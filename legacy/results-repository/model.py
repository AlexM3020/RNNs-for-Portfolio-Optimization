import torch.nn as nn

class PortfolioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_assets=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_assets)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])  # Use final hidden state
        weights = self.softmax(out)  # Portfolio weights sum to 1
        return weights
