import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Cargar datos
df = pd.read_csv("tool_evaluations_10.csv")

def plot_metric_bars(
    df: pd.DataFrame,
    metric_columns: List[str],
    group_col: str = "tool_name",
    title: str = "",
    ylabel: str = "",
    yunit: str = "",
    legend_labels: List[str] = None,
    color: str = "orange"
):
    grouped = df.groupby(group_col)[metric_columns].agg(['mean', 'std'])
    tools = grouped.index.tolist()
    num_metrics = len(metric_columns)
    bar_width = 0.8 / num_metrics
    x = range(len(tools))

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metric_columns):
        means = grouped[(metric, 'mean')]
        stds = grouped[(metric, 'std')]
        x_pos = [val + i * bar_width for val in x]
        label = legend_labels[i] if legend_labels else metric
        plt.bar(x_pos, means, yerr=stds, width=bar_width, capsize=5, label=label, color=color)

    plt.title(title)
    plt.ylabel(f"{ylabel} ({yunit})" if yunit else ylabel)
    plt.xlabel("Tool Name")
    plt.xticks([val + bar_width * (num_metrics-1) / 2 for val in x], tools, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 1. Energía (total, cpu, gpu, ram)
plot_metric_bars(
    df,
    metric_columns=["total_energy", "cpu_energy", "gpu_energy", "ram_energy"],
    title="Energy Consumption by Tool",
    ylabel="Energy",
    yunit="kwH",
    legend_labels=["Total", "CPU", "GPU", "RAM"]
)

# 2. Utilización (CPU y GPU)
plot_metric_bars(
    df,
    metric_columns=["cpu_utilization", "gpu_utilization"],
    title="Hardware Utilization by Tool",
    ylabel="Utilization",
    yunit="%",
    legend_labels=["CPU", "GPU"]
)

# 3. Tiempo de ejecución
plot_metric_bars(
    df,
    metric_columns=["execution_seconds"],
    title="Execution Time by Tool",
    ylabel="Time",
    yunit="Seconds",
    legend_labels=["Execution Time"]
)
