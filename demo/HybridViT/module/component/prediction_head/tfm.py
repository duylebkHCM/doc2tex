import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor
from .addon_module import WordPosEnc
from ...converter.tfm_converter import TFMLabelConverter as TFM
from ....beam import Beam


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.TransformerDecoder:
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return decoder


class TransformerPrediction(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int,
        max_seq_len: int,
        padding_idx: int,
        device: str = "cuda:1",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.num_classes = num_classes
        self.device = device
        self.word_embed = nn.Embedding(num_classes, d_model, padding_idx=padding_idx)

        self.pos_enc = WordPosEnc(d_model=d_model)
        self.d_model = d_model
        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.proj = nn.Linear(d_model, num_classes)
        self.beam = Beam(
            ignore_w=TFM.PAD(),
            start_w=TFM.START(),
            stop_w=TFM.END(),
            max_len=self.max_seq_len,
            device=self.device,
        )

    def reset_beam(self):
        self.beam = Beam(
            ignore_w=TFM.PAD(),
            start_w=TFM.START(),
            stop_w=TFM.END(),
            max_len=self.max_seq_len,
            device=self.device,
        )

    def _build_attention_mask(self, length):
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask = torch.triu(mask).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _embedd_tgt(self, tgt: LongTensor, tgt_len: int):
        tgt_mask = self._build_attention_mask(tgt_len)
        if self.training:
            tgt_pad_mask = tgt == self.padding_idx
        else:
            tgt_pad_mask = None
        tgt = self.word_embed(tgt)
        tgt = self.pos_enc(tgt * math.sqrt(self.d_model))
        return tgt, tgt_mask, tgt_pad_mask

    def forward_greedy(
        self,
        src: FloatTensor,
        tgt: LongTensor,
        output_weight: bool = False,
        is_test: bool = False,
    ) -> FloatTensor:
        if self.training:
            _, l = tgt.size()
            tgt, tgt_mask, tgt_pad_mask = self._embedd_tgt(tgt, l)

            src = rearrange(src, "b t d -> t b d")
            tgt = rearrange(tgt, "b l d -> l b d")

            out = self.model(
                tgt=tgt,
                memory=src,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )

            out = rearrange(out, "l b d -> b l d")
            out = self.proj(out)
        else:
            out = None
            src = rearrange(src, "b t d -> t b d")

            end_flag = torch.zeros(src.shape[0], dtype=torch.bool, device=self.device)

            for step in range(self.max_seq_len + 1):
                b, l = tgt.size()
                emb_tgt, tgt_mask, tgt_pad_mask = self._embedd_tgt(tgt, l)
                emb_tgt = rearrange(emb_tgt, "b l d -> l b d")

                out = self.model(tgt=emb_tgt, memory=src, tgt_mask=tgt_mask)

                out = rearrange(out, "l b d -> b l d")
                out = self.proj(out)
                probs = F.softmax(out, dim=-1)
                next_text = torch.argmax(probs[:, -1:, :], dim=-1)
                tgt = torch.cat([tgt, next_text], dim=-1)

                end_flag = end_flag | (next_text == TFM.END())
                if end_flag.all() and is_test:
                    break

        _, preds_index = out.max(dim=2)
        return preds_index, out

    def forward_beam(self, src: torch.FloatTensor, beam_size: int):
        assert (
            src.size(0) == 1
        ), f"beam search should only have signle source, encounter with batch size: {src.size(0)}"
        out = None
        src = src.squeeze(0)

        for step in range(self.max_seq_len + 1):
            hypotheses = self.beam.hypotheses
            hyp_num = hypotheses.size(0)
            l = hypotheses.size(1)
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"

            emb_tgt = self.word_embed(hypotheses)
            emb_tgt = self.pos_enc(emb_tgt * math.sqrt(self.d_model))
            tgt_mask = self._build_attention_mask(l)
            emb_tgt = rearrange(emb_tgt, "b l d -> l b d")

            exp_src = repeat(src.squeeze(1), "s e -> s b e", b=hyp_num)

            out = self.model(tgt=emb_tgt, memory=exp_src, tgt_mask=tgt_mask)

            out = rearrange(out, "l b d -> b l d")
            out = self.proj(out)
            log_prob = F.log_softmax(out[:, step, :], dim=-1)
            new_hypotheses, new_hyp_scores = self.beam.advance(
                log_prob, step, beam_size=beam_size
            )

            if self.beam.done(beam_size):
                break

            self.beam.set_current_state(new_hypotheses)
            self.beam.set_current_score(new_hyp_scores)

        self.beam.set_hypothesis()
        best_hyp = max(self.beam.completed_hypotheses, key=lambda h: h.score / len(h))
        output = best_hyp.seq
        output = torch.LongTensor(output).unsqueeze(0)
        score = best_hyp.score

        return output, score

    def forward(self, beam_size, batch_H, text, is_test):
        if self.training:
            return self.forward_greedy(batch_H, text)
        else:
            if beam_size > 1:
                return self.forward_beam(batch_H, beam_size)
            else:
                return self.forward_greedy(batch_H, text, is_test=is_test)
