import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def compute_mean_std_table(
    df: pd.DataFrame,
    metric_columns: List[str],
    group_col: str = "tool_name",
    title: str = "",
    legend_labels: List[str] = None,
    users: int = None,
    nodes: int = None,
    export_csv_path: str = None,
    max_cv: float = 1.0
) -> pd.DataFrame:
    df_long = df[[group_col] + metric_columns].melt(
        id_vars=[group_col],
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Value"
    ).dropna(subset=["Value"])

    if legend_labels:
        metric_map = dict(zip(metric_columns, legend_labels))
        df_long["Metric"] = df_long["Metric"].map(metric_map)

    summary_df = (
        df_long
        .groupby([group_col, "Metric"], observed=True)
        .agg(mean_value=("Value", "mean"), std_value=("Value", "std"))
        .reset_index()
    )

    summary_df["mean_value_safe"] = summary_df["mean_value"].replace(0, np.nan)
    summary_df["cv"] = summary_df["std_value"] / summary_df["mean_value_safe"]
    condition = (summary_df["cv"] > max_cv) | (summary_df["cv"].isna()) | (summary_df["cv"] == np.inf)
    summary_df.loc[condition, "std_value"] = summary_df.loc[condition, "mean_value"] * max_cv
    summary_df["std_value"] = summary_df["std_value"].fillna(0).clip(lower=0)
    summary_df.drop(columns=["mean_value_safe"], inplace=True)

    if users is not None:
        summary_df["users"] = users
    if nodes is not None:
        summary_df["nodes"] = nodes

    if export_csv_path:
        summary_df.to_csv(export_csv_path, index=False)

    print(summary_df.groupby([group_col, "Metric"]).size())
    print(f"[INFO] Tabla generada: {title} con users={users}, nodes={nodes}, max_cv={max_cv}")
    return summary_df

def plot_metric_bars_from_summary(
    summary_df: pd.DataFrame,
    group_col: str = "tool_name",
    title: str = "",
    ylabel: str = "",
    yunit: str = "",
    tool_order: List[str] = None,
    metric_order: List[str] = None
):
    sns.set(style="whitegrid")
    pastel = sns.color_palette("pastel")

    if tool_order:
        summary_df[group_col] = pd.Categorical(summary_df[group_col], categories=tool_order, ordered=True)
    if metric_order:
        summary_df["Metric"] = pd.Categorical(summary_df["Metric"], categories=metric_order, ordered=True)
    else:
        summary_df["Metric"] = pd.Categorical(summary_df["Metric"], ordered=True)

    mean_pivot = summary_df.pivot(index=group_col, columns="Metric", values="mean_value")
    std_pivot = summary_df.pivot(index=group_col, columns="Metric", values="std_value")

    if tool_order:
        mean_pivot = mean_pivot.loc[tool_order]
        std_pivot = std_pivot.loc[tool_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    n_metrics = len(mean_pivot.columns)
    bar_width = 0.8 / n_metrics
    x = np.arange(len(mean_pivot.index))
    colors = pastel[:n_metrics]

    for i, metric in enumerate(mean_pivot.columns):
        offset = (i - (n_metrics - 1) / 2) * bar_width
        pos = x + offset
        means = mean_pivot[metric]
        stds = std_pivot[metric]

        ax.bar(pos, means, bar_width, label=str(metric), color=colors[i], edgecolor=None, zorder=2)
        ax.errorbar(pos, means, yerr=stds, fmt='none', ecolor='black', elinewidth=1.3, capsize=0, zorder=3)

    # Estética
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel(f"{ylabel} ({yunit})" if yunit else ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(mean_pivot.index, rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    ax.legend(
        title="Metric", title_fontsize=9, fontsize=8,
        loc='upper right', bbox_to_anchor=(1, 1),
        frameon=True, edgecolor="black"
    )

    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    plt.show()


# -----------------------------------
# DATOS Y USO
# -----------------------------------
df1 = pd.read_csv("3Hour/20_3/tool_evaluations_3.csv")
df1["Scenario"] = "3"

df2 = pd.read_csv("3Hour/20_5/tool_evaluations_5.csv")
df2["Scenario"] = "5"

df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined["group"] = df_combined["tool_name"] + " (" + df_combined["Scenario"] + ")"

# Orden de tools y etiquetas para eje X
tools = ["decide_tool", "lookup_sales_data", "analyzing_data", "create_visualization"]
tools2 = ["lookup_sales_data", "analyzing_data", "create_visualization"]
scenarios = ["3", "5"]
tool_order = [f"{tool} ({sc})" for tool in tools for sc in scenarios]
tool_order2 = [f"{tool} ({sc})" for tool in tools2 for sc in scenarios]

# -----------------------------------
# ENERGY
# -----------------------------------
summary = compute_mean_std_table(
    df_combined,
    metric_columns=["total_energy", "cpu_energy", "gpu_energy", "ram_energy"],
    group_col="group",
    title="Energy Comparison",
    legend_labels=["Total", "CPU", "GPU", "RAM"],
    max_cv=0.9
)

plot_metric_bars_from_summary(
    summary,
    group_col="group",
    title="20 users with 3 and 5 nodes",
    ylabel="Energy",
    yunit="kWh",
    tool_order=tool_order,
    metric_order=["Total", "CPU", "GPU", "RAM"]
)

# -----------------------------------
# UTILIZATION
# -----------------------------------
summary2 = compute_mean_std_table(
    df_combined,
    metric_columns=["cpu_utilization", "gpu_utilization"],
    group_col="group",
    title="Utilization Comparison",
    legend_labels=["CPU", "GPU"],
    max_cv=0.9
)

plot_metric_bars_from_summary(
    summary2,
    group_col="group",
    title="20 users with 3 and 5 nodes",
    ylabel="Utilization",
    yunit="%",
    tool_order=tool_order2,
    metric_order=["CPU", "GPU"]
)

# -----------------------------------
# EXECUTION TIME
# -----------------------------------
summary3 = compute_mean_std_table(
    df_combined,
    metric_columns=["execution_time"],
    group_col="group",
    title="Service Time Comparison",
    legend_labels=["Service Time"],
    max_cv=0.9
)

plot_metric_bars_from_summary(
    summary3,
    group_col="group",
    title="20 users with 3 and 5 nodes",
    ylabel="Time",
    yunit="Seconds",
    tool_order=tool_order,
    metric_order=["Service Time"]
)


