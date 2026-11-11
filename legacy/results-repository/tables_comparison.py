import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Abbreviated metric names
metric_map = {
    "Expected Return (Annualized)": "Exp. Ret.",
    "Volatility (Annualized)": "Volatility",
    "Sharpe Ratio": "Sharpe",
    "Sortino Ratio": "Sortino",
    "Maximum Drawdown": "Max DD",
    "% Positive Returns": "% Pos",
    "Avg Gain/Avg Loss": "AvgG/L",
    "Max Log Cumulative Sum": "Max LogCum"
}

# Raw data
data = {
    "Metric": list(metric_map.keys()),
    "Deep Learning": [0.1641, 0.0988, 1.6607, 2.2263, -0.1490, 55.23, 1.148, 2.3730],
    "Minimum Variance": [0.1506, 0.1016, 1.4822, 2.3136, -0.2293, 53.91, 1.0651, 2.1593],
    "Fixed Allocation 1": [0.0967, 0.1018, 0.9498, 1.9059, -0.0764, 46.38, 1.3355, 1.3897],
    "Fixed Allocation 2": [0.1068, 0.1012, 1.0552, 1.9303, -0.1185, 49.27, 1.2046, 1.5375],
    "Mean Variance": [0.0320, 0.1022, 0.3134, 0.5338, -0.2308, 46.23, 1.1714, 0.5591]
}

# DataFrame transformation
df = pd.DataFrame(data)
df["Metric"] = df["Metric"].map(metric_map)
df.set_index("Metric", inplace=True)
df = df.T

# Metrics to minimize
minimize_metrics = ["Volatility", "Max DD"]

# Format and highlight logic
def format_val(val):
    return f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"

def get_cell_colors(df):
    colors = np.full(df.shape, 'white', dtype=object)
    for col_idx, col in enumerate(df.columns):
        if col == "Max DD":
            # Per Max DD: il valore meno negativo è migliore
            best_idx = df[col].idxmax()  # meno negativo
            worst_idx = df[col].idxmin()  # più negativo
        elif col in minimize_metrics:
            best_idx = df[col].idxmin()
            worst_idx = df[col].idxmax()
        else:
            best_idx = df[col].idxmax()
            worst_idx = df[col].idxmin()

        for row_idx, idx in enumerate(df.index):
            if idx == best_idx:
                colors[row_idx, col_idx] = '#a1d99b'  # verde
            elif idx == worst_idx:
                colors[row_idx, col_idx] = '#ff9f9b'  # rosso
    return colors

# Apply formatting
formatted_text = df.applymap(format_val)
cell_colors = get_cell_colors(df)

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))  # Smaller width
ax.axis('off')

table = ax.table(
    cellText=formatted_text.values,
    rowLabels=formatted_text.index,
    colLabels=formatted_text.columns,
    cellColours=cell_colors,
    colColours=['#f0f0f0'] * df.shape[1],
    cellLoc='center',
    loc='center'
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.4, 1.4)  # More compact

# Header and row label styling
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    if col == -1 and row > 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f0')

plt.title("Strategy Performance Comparison (Volatility-Scaled)", pad=20, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('strategy_comparison_compact.png', dpi=300, bbox_inches='tight')
plt.show()
