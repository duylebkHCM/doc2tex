import torch.nn as nn

__all__ = ["BiLSTM_Seq_Modeling", "BidirectionalLSTM"]


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class BiLSTM_Seq_Modeling(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(BiLSTM_Seq_Modeling, self).__init__()
        self.num_layers = num_layers
        layers = []
        layers += [BidirectionalLSTM(input_size, hidden_size, hidden_size)]
        for i in range(num_layers - 2):
            layers.append(BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        layers.append(BidirectionalLSTM(hidden_size, hidden_size, output_size))
        self.lstm = nn.Sequential(*layers)

    def forward(self, input):
        return self.lstm(input)
