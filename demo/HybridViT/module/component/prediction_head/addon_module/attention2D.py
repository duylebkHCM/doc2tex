import torch
import torch.nn as nn
from .attention1D import LocationAwareAttention

"""
NOTE :
"""


class SARAttention(nn.Module):
    def __init__(
        self,
        input_size,
        attention_size,
        backbone_size,
        output_size,
    ):
        self.conv1x1_1 = nn.Conv2d(output_size, attention_size, kernel_size=1, stride=1)
        self.conv3x3 = nn.Conv2d(
            backbone_size, attention_size, kernel_size=3, stride=1, padding=1
        )
        self.conv1x1_2 = nn.Conv2d(attention_size, 1, kernel_size=1, stride=1)

        self.rnn_decoder_1 = nn.LSTMCell(input_size, input_size)
        self.rnn_decoder_2 = nn.LSTMCell(input_size, input_size)

    def forward(
        self,
        dec_input,
        feature_map,
        holistic_feature,
        hidden_1,
        cell_1,
        hidden_2,
        cell_2,
    ):
        _, _, H_feat, W_feat = feature_map.size()
        hidden_1, cell_1 = self.rnn_decoder_1(dec_input, (hidden_1, cell_1))
        hidden_2, cell_2 = self.rnn_decoder_2(hidden_1, (hidden_2, cell_2))

        hidden_2_tile = hidden_2.view(hidden_2.size(0), hidden_2.size(1), 1, 1)
        attn_query = self.conv1x1_1(hidden_2_tile)
        attn_query = attn_query.expand(-1, -1, H_feat, W_feat)

        attn_key = self.conv3x3(feature_map)
        attn_weight = torch.tanh(torch.add(attn_query, attn_key, alpha=1))
        attn_weight = self.conv1x1_2(attn_weight)  # shape B, 1, H, W

        # TO DO: apply mask for attention weight


class LocationAwareAttentionCell2D(nn.Module):
    def __init__(self, kernel_size, kernel_dim, hidden_dim, input_dim):
        super().__init__()
        self.loc_conv = nn.Conv2d(
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
        batch_size, enc_h, enc_w, hidden_dim = (
            encoder_output.shape[0],
            encoder_output.shape[1],
            encoder_output.shape[2],
            decoder_hidden[0].shape[1],
        )

        encoder_proj = self.key_proj(encoder_output)
        hidden_proj = self.query_proj(decoder_hidden[0]).unsqueeze(1)

        if last_alignment is None:
            last_alignment = decoder_hidden[0].new_zeros(batch_size, enc_h, enc_w, 1)

        loc_context = self.loc_conv(last_alignment.permute(0, 2, 1))
        loc_context = loc_context.transpose(1, 2)
        loc_context = self.loc_proj(loc_context)

        assert len(loc_context.shape) == 3
        assert (
            loc_context.shape[0] == batch_size
        ), f"{loc_context.shape[0]}-{batch_size}"
        assert loc_context.shape[1] == enc_h
        assert loc_context.shape[2] == enc_w
        assert loc_context.shape[3] == hidden_dim

        loc_context = loc_context.reshape(batch_size, enc_h * enc_w, hidden_dim)

        score = self.score(torch.tanh(encoder_proj + hidden_proj + loc_context))
        return score


class LocationAwareAttention2D(LocationAwareAttention):
    def __init__(
        self, kernel_size, kernel_dim, temperature=1, smoothing=False, *args, **kwargs
    ):
        super().__init__(
            kernel_size, kernel_dim, temperature, smoothing, *args, **kwargs
        )
        self.attn = LocationAwareAttentionCell2D(
            kernel_size, kernel_dim, self.hidden_size, self.input_size
        )
