import torch
from typing import List
from einops import rearrange, repeat
from typing import Optional


class Hypothesis:
    seq: List[int]
    score: float
    attn_weights: List[float]

    def __init__(
        self,
        seq_tensor: torch.LongTensor,
        score: float,
        weights: Optional[torch.FloatTensor] = None,
    ) -> None:
        raw_seq = seq_tensor.tolist()

        self.seq = raw_seq
        self.score = score
        if weights:
            self.attn_weights = weights.tolist()
            assert len(self.attn_weights) == len(self.seq)
        else:
            self.attn_weights = None

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}, weight: {self.attn_weights}"


class Beam:
    def __init__(
        self,
        start_w=1,
        stop_w=2,
        ignore_w=0,
        max_len=150,
        viz_attn=False,
        device="cuda",
    ):
        self.stop_w = stop_w
        self.start_w = start_w

        self.hypotheses = torch.full(
            (1, max_len + 2),
            fill_value=ignore_w,
            dtype=torch.long,
            device=device,
        )
        if viz_attn:
            self.hyp_alpha = torch.ones(
                1, max_len + 2, dtype=torch.float, device=device
            )

        self.hypotheses[:, 0] = start_w
        self.hyp_scores = torch.zeros(1, dtype=torch.float, device=device)
        self.completed_hypotheses: List[Hypothesis] = []
        self.device = device
        self.viz_attn = viz_attn

    def advance(self, next_log_probs, step, beam_size):
        vocab_size = next_log_probs.shape[1]
        live_hyp_num = beam_size - len(self.completed_hypotheses)
        exp_hyp_scores = repeat(self.hyp_scores, "b -> b e", e=vocab_size)
        continuous_hyp_scores = rearrange(
            exp_hyp_scores + next_log_probs, "b e -> (b e)"
        )
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
            continuous_hyp_scores, k=live_hyp_num
        )

        prev_hyp_ids = top_cand_hyp_pos // vocab_size
        hyp_word_ids = top_cand_hyp_pos % vocab_size

        step += 1
        new_hypotheses = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
            prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
        ):
            cand_new_hyp_score = cand_new_hyp_score.detach().item()
            self.hypotheses[prev_hyp_id, step] = hyp_word_id

            if hyp_word_id == self.stop_w:
                self.completed_hypotheses.append(
                    Hypothesis(
                        seq_tensor=self.hypotheses[prev_hyp_id, 1 : step + 1]
                        .detach()
                        .clone(),  # remove START_W at first
                        score=cand_new_hyp_score,
                    )
                )
            else:
                new_hypotheses.append(self.hypotheses[prev_hyp_id].detach().clone())
                new_hyp_scores.append(cand_new_hyp_score)

        return new_hypotheses, new_hyp_scores

    def get_incomplete_inds(self, hyp_word_ids):
        return [
            ind
            for ind, next_word in enumerate(hyp_word_ids)
            if next_word != self.stop_w
        ]

    def get_complete_inds(self, hyp_word_ids, incomplete_inds):
        return list(set(range(len(hyp_word_ids))) - set(incomplete_inds))

    def set_current_state(self, hypotheses):
        "Set the outputs for the current timestep."
        self.hypotheses = torch.stack(hypotheses, dim=0)
        return

    def set_current_score(self, hyp_scores):
        "Set the scores for the current timestep."
        self.hyp_scores = torch.tensor(
            hyp_scores, dtype=torch.float, device=self.device
        )
        return

    def done(self, beam_size):
        return len(self.completed_hypotheses) == beam_size

    def set_hypothesis(self):
        if len(self.completed_hypotheses) == 0:
            self.completed_hypotheses.append(
                Hypothesis(
                    seq_tensor=self.hypotheses[0, 1:].detach().clone(),
                    score=self.hyp_scores[0].detach().item(),
                )
            )
        return
