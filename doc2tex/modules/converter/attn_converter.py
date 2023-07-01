import torch
import numpy as np


class AttnLabelConverter(object):
    """Convert between text-label and text-index"""

    list_token = ["[GO]", "[s]", "[UNK]"]

    def __init__(self, character, device):
        list_character = character
        self.character = AttnLabelConverter.list_token + list_character

        self.device = device
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        self.ignore_idx = self.dict["[GO]"]

    @staticmethod
    def START() -> int:
        return AttnLabelConverter.list_token.index("[GO]")

    @staticmethod
    def END() -> int:
        return AttnLabelConverter.list_token.index("[s]")

    @staticmethod
    def UNK() -> int:
        return AttnLabelConverter.list_token.index("[UNK]")

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)

            if len(text) > batch_max_length:
                text = text[: (batch_max_length - 1)]

            text.append("[s]")
            text = [
                self.dict[char] if char in self.dict else self.dict["[UNK]"]
                for char in text
            ]

            batch_text[i][1 : 1 + len(text)] = torch.LongTensor(
                text
            )  # batch_text[:, 0] = [GO] token
        return (batch_text.to(self.device), torch.IntTensor(length).to(self.device))

    def decode(self, text_index, token_level="word"):
        """convert text-index into text-label."""
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            if token_level == "word":
                text = " ".join([self.character[i] for i in text_index[index, :]])
            else:
                text = "".join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

    def detokenize(self, token_ids):
        """convert token ids to list of token"""
        b_toks = []
        for tok in token_ids:
            toks = []
            for i in tok:
                if self.character[i] == "[s]":
                    break
                toks.append(self.character[i])
            b_toks.append(toks)

        return b_toks
