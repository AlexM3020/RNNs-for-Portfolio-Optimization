from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
from model import PortfolioLSTM
from sharp_ratio import sharpe_ratio_loss
import numpy as np
import pandas as pd
from datetime import datetime

# --------------------------
# Define rolling windows
# --------------------------
rolling_windows = [
    ('2010-01-01', '2012-01-01', '2012-01-01', '2014-01-01'),
    ('2010-01-01', '2014-01-01', '2014-01-01', '2016-01-01'),
    ('2010-01-01', '2016-01-01', '2016-01-01', '2018-01-01'),
    ('2010-01-01', '2018-01-01', '2018-01-01', '2020-01-01'),
    ('2010-01-01', '2020-01-01', '2020-01-01', '2022-01-01'),
    ('2010-01-01', '2022-01-01', '2022-01-01', '2024-01-01'),
    ('2010-01-01', '2024-01-01', '2024-01-01', '2024-12-31'),  # test finale
]

# --------------------------
# Load preprocessed data
# --------------------------
data = torch.load('preprocessed_data.pt', weights_only=False)
returns_df = data['returns']  # pandas DataFrame: daily asset returns
X_tensor = data['X_tensor']   # torch.Tensor: input sequences

print("Loaded X_tensor:", X_tensor.shape)
print("Loaded returns_df:", returns_df.shape)

# --------------------------
# Align and normalize returns
# --------------------------
aligned_returns = returns_df.shift(-1).iloc[50:]
aligned_returns.dropna(inplace=True)

min_len = min(len(X_tensor), len(aligned_returns))
X_tensor = X_tensor[:min_len]
aligned_returns = aligned_returns.iloc[:min_len]

normalized_returns = (aligned_returns - aligned_returns.mean()) / (aligned_returns.std() + 1e-8)
future_returns = torch.tensor(normalized_returns.values, dtype=torch.float32)

torch.save({
    'X_tensor': X_tensor,
    'aligned_returns': aligned_returns,
    'future_returns': future_returns,
}, f'data_windows/data1')

# --------------------------
# Get date index for slicing
# --------------------------
dates = aligned_returns.index
assert len(dates) == len(X_tensor), "Date index and tensor mismatch"

# --------------------------
# Rolling training loop
# --------------------------
all_val_sharpe = []

for i, (train_start, train_end, test_start, test_end) in enumerate(rolling_windows):
    print(f"\nWindow {i+1}: Train {train_start} → {train_end}, Test {test_start} → {test_end}")

    # Get index masks
    train_mask = (dates >= train_start) & (dates < train_end)
    test_mask = (dates >= test_start) & (dates < test_end)

    # Extract slices
    X_train = X_tensor[train_mask]
    y_train = future_returns[train_mask]
    X_test = X_tensor[test_mask]
    y_test = future_returns[test_mask]

    if len(X_train) < 100 or len(X_test) < 100:
        print("Skipping window due to insufficient data")
        continue

    # Wrap into datasets
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    # Init model
    model = PortfolioLSTM(input_size=8)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5,
                                                          min_lr=1e-6)

    # Training loop
    best_loss = float('inf')
    patience = 20
    no_improve = 0
    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            weights = model(xb)
            weights = torch.clamp(weights, 0.0, 1.0)
            loss = sharpe_ratio_loss(weights, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in test_loader:
                weights = model(xb)
                val_loss = sharpe_ratio_loss(weights, yb)
                val_losses.append(val_loss.item())
            avg_loss = np.mean(val_losses)
            print(f"Epoch {epoch+1}: Val Sharpe Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0

            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping")
                    break

    all_val_sharpe.append((test_start, test_end, best_loss))

# --------------------------
# Final summary
# --------------------------
print("\nRolling Sharpe Loss Summary:")
for start, end, loss in all_val_sharpe:
    print(f"{start} → {end} : {loss:.4f}")
# --------------------------
# Save the final model after all windows
# --------------------------
torch.save(model.state_dict(), "models/final_lstm_model_weights7.pth")
torch.save(model, "models/final_lstm_full_model7.pt")

