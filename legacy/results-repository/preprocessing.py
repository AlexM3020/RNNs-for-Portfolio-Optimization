import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
def create_sequences(features, window=50):
    """
This function creates sliding windows of size window (default = 50).
For example:
If your features have 300 days of data, you get sequences like:
Day 0–49 → input 1
Day 1–50 → input 2
...
Day 249–298 → input 250
Each window becomes a training sample. This is critical for LSTM which expects sequences as input.
    """
    X = []
    for i in range(window, len(features)):
        # Stack prices and returns: [price_1, price_2, ..., return_1, return_2, ...]
        x = features.iloc[i-window:i].values
        X.append(x)
    return np.array(X)
# Load treturns from CSV
returns = pd.read_csv("returns.csv", index_col=0, header=[0, 1])  # MultiIndex columns
# Load the features from CSV
features = pd.read_csv("etf_features.csv", index_col=0, header=[0, 1])  # MultiIndex columns
# Output: 3D NumPy array with shape [num_sequences, window, num_features]
seq_data = create_sequences(features)

# Normalize for stability, z-score normalization
seq_data = (seq_data - seq_data.mean(axis=(0,1))) / seq_data.std(axis=(0,1))


# Converts the NumPy array into a torch.Tensor for use in training
X_tensor = torch.tensor(seq_data, dtype=torch.float32)

# After your existing preprocessing code:
torch.save({
    'features': features,
    'returns': returns,
    'X_tensor': X_tensor
}, 'preprocessed_data.pt')