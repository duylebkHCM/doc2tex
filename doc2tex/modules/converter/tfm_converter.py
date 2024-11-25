import torch
import numpy as np


class TFMLabelConverter(object):
    """Convert between text-label and text-index"""

    list_token = ["[PAD]", "[GO]", "[s]", "[UNK]"]

    def __init__(self, character, device):
        list_character = character
        self.character = TFMLabelConverter.list_token + list_character

        self.device = device
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        self.ignore_idx = self.dict["[PAD]"]

    @staticmethod
    def START() -> int:
        return TFMLabelConverter.list_token.index("[GO]")

    @staticmethod
    def END() -> int:
        return TFMLabelConverter.list_token.index("[s]")

    @staticmethod
    def UNK() -> int:
        return TFMLabelConverter.list_token.index("[UNK]")

    @staticmethod
    def PAD() -> int:
        return TFMLabelConverter.list_token.index("[PAD]")

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_max_length += 1
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(
            self.ignore_idx
        )
        for i, t in enumerate(text):
            text = list(t)

            if len(text) > batch_max_length:
                text = text[: (batch_max_length - 1)]

            text.append("[s]")
            text = [
                self.dict[char] if char in self.dict else self.dict["[UNK]"]
                for char in text
            ]
            batch_text[i][0] = torch.LongTensor([self.dict["[GO]"]])
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


if __name__ == "__main__":
    vocab = [
        "S",
        "ố",
        " ",
        "2",
        "5",
        "3",
        "đ",
        "ư",
        "ờ",
        "n",
        "g",
        "T",
        "r",
        "ầ",
        "P",
        "h",
        "ú",
        ",",
        "ị",
        "t",
        "ấ",
        "N",
        "a",
        "m",
        "á",
        "c",
        "H",
        "u",
        "y",
        "ệ",
        "ả",
        "i",
        "D",
        "ơ",
        "8",
        "9",
        "Đ",
        "B",
        "ộ",
        "L",
        "ĩ",
        "6",
        "Q",
        "ậ",
        "ì",
        "ạ",
        "ồ",
        "C",
        "í",
        "M",
        "4",
        "E",
        "/",
        "K",
        "p",
        "1",
        "A",
        "x",
        "ặ",
        "ễ",
        "0",
        "â",
        "à",
        "ế",
        "ừ",
        "ê",
        "-",
        "7",
        "o",
        "V",
        "ô",
        "ã",
        "G",
        "ớ",
        "Y",
        "I",
        "ề",
        "ò",
        "l",
        "R",
        "ỹ",
        "ủ",
        "X",
        "'",
        "e",
        "ắ",
        "ổ",
        "ằ",
        "k",
        "s",
        ".",
        "ợ",
        "ù",
        "ứ",
        "ă",
        "ỳ",
        "ẵ",
        "ý",
        "ó",
        "ẩ",
        "ọ",
        "J",
        "ũ",
        "ữ",
        "ự",
        "õ",
        "ỉ",
        "ỏ",
        "v",
        "d",
        "Â",
        "W",
        "U",
        "O",
        "é",
        "ở",
        "ỷ",
        "(",
        ")",
        "ử",
        "è",
        "ể",
        "ụ",
        "ỗ",
        "F",
        "q",
        "ẻ",
        "ỡ",
        "b",
        "ỵ",
        "Ứ",
        "#",
        "ẽ",
        "Ô",
        "Ê",
        "Ơ",
        "+",
        "z",
        "Ấ",
        "w",
        "Z",
        "&",
        "Á",
        "~",
        "f",
        "Ạ",
        "Ắ",
        "j",
        ":",
        "Ă",
        "<",
        ">",
        "ẹ",
        "_",
        "À",
        "Ị",
        "Ư",
        "Ễ",
    ]
    text = [
        "190B Trần Quang Khải, Phường Tân Định, Quận 1, TP Hồ Chí Minh",
        "164/2B, Quốc lộ 1A, Phường Lê Bình, Quận Cái Răng, Cần Thơ",
        "Cẩm Huy, Huyện Cẩm Xuyên, Hà Tĩnh",
    ]
    tfm_convert = TFMLabelConverter(vocab, "cpu")
    texts, lengths = tfm_convert.encode(text, 70)
    print(texts)
    for text in texts:
        print("Encode", text)
        text = text.unsqueeze(0)
        decode_text = tfm_convert.decode(text, "char")
        print("Decode", decode_text)
