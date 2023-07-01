import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from ...converter import AttnLabelConverter as ATTN
from .addon_module import *


class Attention(nn.Module):
    def __init__(
        self,
        kernel_size,
        kernel_dim,
        input_size,
        hidden_size,
        num_classes,
        embed_dim=None,
        attn_type="coverage",
        embed_target=False,
        enc_init=False,  # init hidden state of decoder with enc output
        teacher_forcing=1.0,
        droprate=0.1,
        method="concat",
        seqmodel="ViT",
        viz_attn: bool = False,
        device="cuda",
    ):
        super(Attention, self).__init__()
        if embed_dim is None:
            embed_dim = input_size
        if embed_target:
            self.embedding = nn.Embedding(
                num_classes, embed_dim, padding_idx=ATTN.START()
            )

        common = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_embeddings": embed_dim if embed_target else num_classes,
            "num_classes": num_classes,
        }

        if attn_type == "luong":
            common["method"] = method
            self.attention_cell = LuongAttention(**common)
        elif attn_type == "loc_aware":
            self.attention_cell = LocationAwareAttention(
                kernel_size=kernel_size, kernel_dim=kernel_dim, **common
            )
        elif attn_type == "coverage":
            self.attention_cell = LocationAwareAttention(
                kernel_size=kernel_size, kernel_dim=kernel_dim, **common
            )
        else:
            self.attention_cell = BahdanauAttention(**common)

        self.dropout = nn.Dropout(droprate)
        self.embed_target = embed_target
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.teacher_forcing = teacher_forcing
        self.device = device
        self.attn_type = attn_type
        self.enc_init = enc_init
        self.viz_attn = viz_attn
        self.seqmodel = seqmodel

        if enc_init:
            self.init_hidden()

    def _embed_text(self, input_char):
        return self.embedding(input_char)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def init_hidden(self):
        self.proj_init_h = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.proj_init_c = nn.Linear(self.input_size, self.hidden_size, bias=True)

    def forward_beam(
        self,
        batch_H: torch.Tensor,
        batch_max_length=25,
        beam_size=4,
    ):
        batch_size = batch_H.size(0)
        assert batch_size == 1
        num_steps = batch_max_length + 1
        batch_H = batch_H.squeeze(dim=0)
        batch_H = repeat(batch_H, "s e -> b s e", b=beam_size)

        if self.enc_init:
            if self.seqmodel == "BiLSTM":
                init_embedding = batch_H.mean(dim=1)
            else:
                init_embedding = batch_H[:, 0, :]
            h_0 = self.proj_init_h(init_embedding)
            c_0 = self.proj_init_c(init_embedding)
            hidden = (h_0, c_0)
        else:
            hidden = (
                torch.zeros(
                    beam_size, self.hidden_size, dtype=torch.float32, device=self.device
                ),
                torch.zeros(
                    beam_size, self.hidden_size, dtype=torch.float32, device=self.device
                ),
            )

        if self.attn_type == "coverage":
            alpha_cum = torch.zeros(
                beam_size, batch_H.shape[1], 1, dtype=torch.float32, device=self.device
            )
        self.attention_cell.reset_mem()

        k_prev_words = torch.LongTensor([[ATTN.START()]] * beam_size).to(self.device)
        seqs = k_prev_words
        targets = k_prev_words.squeeze(dim=-1)
        top_k_scores = torch.zeros(beam_size, 1).to(self.device)

        if self.viz_attn:
            seqs_alpha = torch.ones(beam_size, 1, batch_H.shape[1]).to(self.device)

        complete_seqs = list()
        if self.viz_attn:
            complete_seqs_alpha = list()
        complete_seqs_scores = list()

        for step in range(num_steps):
            embed_text = (
                self._char_to_onehot(targets, onehot_dim=self.num_classes)
                if not self.embed_target
                else self._embed_text(targets)
            )
            output, hidden, alpha = self.attention_cell(hidden, batch_H, embed_text)
            output = self.dropout(output)
            vocab_size = output.shape[1]

            scores = F.log_softmax(output, dim=-1)
            scores = top_k_scores.expand_as(scores) + scores
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(
                    beam_size, 0, True, True
                )

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            if self.viz_attn:
                seqs_alpha = torch.cat(
                    [
                        seqs_alpha[prev_word_inds],
                        alpha[prev_word_inds].permute(0, 2, 1),
                    ],
                    dim=1,
                )

            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != ATTN.END()
            ]

            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                if self.viz_attn:
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds])
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            beam_size = beam_size - len(complete_inds)
            if beam_size == 0:
                break

            seqs = seqs[incomplete_inds]
            if self.viz_attn:
                seqs_alpha = seqs_alpha[incomplete_inds]
            hidden = (
                hidden[0][prev_word_inds[incomplete_inds]],
                hidden[1][prev_word_inds[incomplete_inds]],
            )
            batch_H = batch_H[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            targets = next_word_inds[incomplete_inds]

            if self.attn_type == "coverage":
                alpha_cum = alpha_cum + alpha
                alpha_cum = alpha_cum[incomplete_inds]
                self.attention_cell.set_mem(alpha_cum)
            elif self.attn_type == "loc_aware":
                self.attention_cell.set_mem(alpha)

        if len(complete_inds) == 0:
            seq = seqs[0][1:].tolist()
            seq = torch.LongTensor(seq).unsqueeze(0)
            score = top_k_scores[0]
            if self.viz_attn:
                alphas = seqs_alpha[0][1:, ...]
                return seq, score, alphas
            else:
                return seq, score, None
        else:
            combine_lst = tuple(zip(complete_seqs, complete_seqs_scores))
            best_ind = combine_lst.index(
                max(combine_lst, key=lambda x: x[1] / len(x[0]))
            )  # https://youtu.be/XXtpJxZBa2c?t=2407
            seq = complete_seqs[best_ind][1:]  # not include [GO] token
            seq = torch.LongTensor(seq).unsqueeze(0)
            score = max(complete_seqs_scores)

            if self.viz_attn:
                alphas = complete_seqs_alpha[best_ind][1:, ...]
                return seq, score, alphas
            else:
                return seq, score, None

    def forward_greedy(
        self, batch_H, text, is_train=True, is_test=False, batch_max_length=25
    ):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1
        if self.enc_init:
            if self.seqmodel == "BiLSTM":
                init_embedding = batch_H.mean(dim=1)
                encoder_hidden = batch_H
            else:
                encoder_hidden = batch_H
                init_embedding = batch_H[:, 0, :]
            h_0 = self.proj_init_h(init_embedding)
            c_0 = self.proj_init_c(init_embedding)
            hidden = (h_0, c_0)
        else:
            encoder_hidden = batch_H
            hidden = (
                torch.zeros(
                    batch_size,
                    self.hidden_size,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.zeros(
                    batch_size,
                    self.hidden_size,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

        targets = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )  # [GO] token
        probs = torch.zeros(
            batch_size,
            num_steps,
            self.num_classes,
            dtype=torch.float32,
            device=self.device,
        )

        if self.viz_attn:
            self.alpha_stores = torch.zeros(
                batch_size,
                num_steps,
                encoder_hidden.shape[1],
                1,
                dtype=torch.float32,
                device=self.device,
            )
        if self.attn_type == "coverage":
            alpha_cum = torch.zeros(
                batch_size,
                encoder_hidden.shape[1],
                1,
                dtype=torch.float32,
                device=self.device,
            )

        self.attention_cell.reset_mem()

        if is_test:
            end_flag = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(num_steps):
            embed_text = (
                self._char_to_onehot(targets, onehot_dim=self.num_classes)
                if not self.embed_target
                else self._embed_text(targets)
            )
            output, hidden, alpha = self.attention_cell(
                hidden, encoder_hidden, embed_text
            )
            output = self.dropout(output)
            if self.viz_attn:
                self.alpha_stores[:, i] = alpha
            if self.attn_type == "coverage":
                alpha_cum = alpha_cum + alpha
                self.attention_cell.set_mem(alpha_cum)
            elif self.attn_type == "loc_aware":
                self.attention_cell.set_mem(alpha)

            probs_step = output
            probs[:, i, :] = probs_step

            if i == num_steps - 1:
                break

            if is_train:
                if self.teacher_forcing < random.random():
                    _, next_input = probs_step.max(1)
                    targets = next_input
                else:
                    targets = text[:, i + 1]
            else:
                _, next_input = probs_step.max(1)
                targets = next_input

                if is_test:
                    end_flag = end_flag | (next_input == ATTN.END())
                    if end_flag.all():
                        break

        _, preds_index = probs.max(2)

        return preds_index, probs, None  # batch_size x num_steps x num_classes

    def forward(
        self, beam_size, batch_H, text, batch_max_length, is_train=True, is_test=False
    ):
        if is_train:
            return self.forward_greedy(
                batch_H, text, is_train, is_test, batch_max_length
            )
        else:
            if beam_size > 1:
                return self.forward_beam(batch_H, batch_max_length, beam_size)
            else:
                return self.forward_greedy(
                    batch_H, text, is_train, is_test, batch_max_length
                )
