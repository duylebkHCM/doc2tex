import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class LuongAttention(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_embeddings, num_classes, method="dot"
    ):
        super(LuongAttention, self).__init__()
        self.attn = LuongAttentionCell(hidden_size, method)
        self.rnn = nn.LSTMCell(num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.generator = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, prev_hidden, batch_H, embed_text):
        hidden = self.rnn(embed_text, prev_hidden)

        e = self.attn(hidden[0], batch_H)
        # print('Shape e', e.shape)
        alpha = F.softmax(e, dim=1)
        # print('Shape al', alpha.shape)

        context = torch.bmm(alpha.unsqueeze(1), batch_H).squeeze(
            1
        )  # batch_size x num_channel
        output = torch.cat(
            [context, hidden[0]], 1
        )  # batch_size x (num_channel + num_embedding)
        output = torch.tanh(output)
        output = self.generator(output)

        return output, hidden, alpha


class LuongAttentionCell(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(LuongAttentionCell, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        # print('shape', decoder_hidden.shape)

        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.permute(0, 2, 1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.permute(0, 2, 1)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
            # print('Shape', out.shape)
            return out.bmm(
                self.weight.unsqueeze(-1).repeat(out.shape[0], 1, 1)
            ).squeeze(-1)


class BahdanauAttentionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BahdanauAttentionCell, self).__init__()
        self.i2h = nn.Linear(input_dim, hidden_dim, bias=False)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_output):
        encoder_proj = self.i2h(encoder_output)
        hidden_proj = self.h2h(decoder_hidden[0]).unsqueeze(1)
        score = self.score(torch.tanh(encoder_proj + hidden_proj))
        return score


class BahdanauAttention(nn.Module):
    def __init__(
        self, input_size=100, hidden_size=256, num_embeddings=10, num_classes=10
    ):
        super(BahdanauAttention, self).__init__()
        self.attn = BahdanauAttentionCell(input_size, hidden_size)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)

    def set_mem(self, prev_attn):
        pass

    def reset_mem(self):
        pass

    def forward(self, prev_hidden, batch_H, embed_text):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        e = self.attn(prev_hidden, batch_H)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(
            1
        )  # batch_size x num_channel
        concat_context = torch.cat(
            [context, embed_text], 1
        )  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        output = self.generator(cur_hidden[0])

        return output, cur_hidden, alpha


class LocationAwareAttentionCell(nn.Module):
    def __init__(self, kernel_size, kernel_dim, hidden_dim, input_dim):
        super().__init__()
        self.loc_conv = nn.Conv1d(
            1,
            kernel_dim,
            kernel_size=2 * kernel_size + 1,
            padding=kernel_size,
            bias=True,
        )
        self.loc_proj = nn.Linear(kernel_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_output, last_alignment):
        batch_size, seq_length, hidden_dim = (
            encoder_output.shape[0],
            encoder_output.shape[1],
            decoder_hidden[0].shape[1],
        )

        encoder_proj = self.key_proj(encoder_output)
        hidden_proj = self.query_proj(decoder_hidden[0]).unsqueeze(1)

        if last_alignment is None:
            last_alignment = decoder_hidden[0].new_zeros(batch_size, seq_length, 1)

        loc_context = self.loc_conv(last_alignment.permute(0, 2, 1))
        loc_context = loc_context.transpose(1, 2)
        loc_context = self.loc_proj(loc_context)

        assert len(loc_context.shape) == 3
        assert (
            loc_context.shape[0] == batch_size
        ), f"{loc_context.shape[0]}-{batch_size}"
        assert loc_context.shape[1] == seq_length
        assert loc_context.shape[2] == hidden_dim

        score = self.score(torch.tanh(encoder_proj + hidden_proj + loc_context))
        return score


class CoverageAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        kernel_dim,
        temperature=1.0,
        smoothing=False,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature
        self.prev_attn = None
        self.attn = LocationAwareAttentionCell(
            kernel_size, kernel_dim, hidden_dim, input_dim
        )

    def set_mem(self, prev_attn):
        self.prev_attn = prev_attn

    def reset_mem(self):
        self.prev_attn = None

    def forward(self, prev_hidden, batch_H):
        e = self.attn(prev_hidden, batch_H, self.prev_attn)

        if self.smoothing:
            e = F.sigmoid(e, dim=1)
            alpha = e.div(e.sum(dim=-1).unsqueeze(dim=-1))
        else:
            e = e / self.temperature
            alpha = F.softmax(e, dim=1)

        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(
            1
        )  # batch_size x num_channel

        return context, alpha


class LocationAwareAttention(BahdanauAttention):
    def __init__(
        self, kernel_size, kernel_dim, temperature=1.0, smoothing=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing
        self.temperature = temperature
        self.prev_attn = None
        self.attn = LocationAwareAttentionCell(
            kernel_size, kernel_dim, self.hidden_size, self.input_size
        )

    def set_mem(self, prev_attn):
        self.prev_attn = prev_attn

    def reset_mem(self):
        self.prev_attn = None

    def forward(self, prev_hidden, batch_H, embed_text):
        e = self.attn(prev_hidden, batch_H, self.prev_attn)

        if self.smoothing:
            e = F.sigmoid(e, dim=1)
            alpha = e.div(e.sum(dim=-1).unsqueeze(dim=-1))
        else:
            e = e / self.temperature
            alpha = F.softmax(e, dim=1)

        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(
            1
        )  # batch_size x num_channel , batch_H: batch_sizexseq_lengthxnum_channel, alpha:
        concat_context = torch.cat(
            [context, embed_text], 1
        )  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        output = self.generator(cur_hidden[0])

        return output, cur_hidden, alpha
