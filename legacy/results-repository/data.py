import yfinance as yf
import pandas as pd
import torch
from sklearn.model_selection import train_test_split



# Define the ETFs and their Yahoo Finance tickers
tickers = ["VTI", "AGG", "DBC", "^VIX"]

# Download data from 2009 to end of 2023
data = yf.download(
    tickers,
    start="2010-01-01",
    end="2024-12-31",
    progress=False
)['Close']  # Only get 'Close' prices

# Rename '^VIX' to 'VIX' for consistency
data.rename(columns={"^VIX": "VIX"}, inplace=True)

# Drop any rows with missing values (optional but useful for modeling)
data.dropna(inplace=True)

# Save to CSV
data.to_csv("etf_data_2010_2024.csv")
print("Data saved to 'etf_data_2010_2024.csv'")
#print(data.head())
# Compute daily returns, pct_change computes fractional change between current and previous element in a time series
returns = data.pct_change().fillna(0)
returns.to_csv("returns.csv")
# Combine close prices and returns into one feature tensor
features = pd.concat([data, returns], axis=1, keys=['price', 'return'])
features.to_csv("etf_features.csv")