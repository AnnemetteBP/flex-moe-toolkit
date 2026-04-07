import seaborn as sns
import matplotlib.pyplot as plt



def plot_expert_heatmap(matrix):

    plt.figure(figsize=(8,4))

    sns.heatmap(matrix, cmap="viridis")

    plt.xlabel("Expert")
    plt.ylabel("Layer")

    plt.title("Layer × Expert Usage")

    plt.show()


def plot_expert_combination_upset(combination_counts, path, title, max_combinations=12):
    """
    Save a lightweight upset-style plot for expert activation combinations.
    `combination_counts` should map tuples like (0, 2, 3) -> count.
    """

    if not combination_counts:
        raise ValueError("No expert combinations were provided.")

    top_items = sorted(
        combination_counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )[:max_combinations]

    all_experts = sorted({expert for combo, _count in top_items for expert in combo})
    expert_to_row = {expert: idx for idx, expert in enumerate(all_experts)}

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.05)
    ax_bar = fig.add_subplot(grid[0])
    ax_matrix = fig.add_subplot(grid[1], sharex=ax_bar)

    x_positions = list(range(len(top_items)))
    counts = [count for _combo, count in top_items]
    labels = ["{" + ",".join(str(expert) for expert in combo) + "}" for combo, _count in top_items]

    ax_bar.bar(x_positions, counts, color="#2f6db2")
    ax_bar.set_ylabel("Examples")
    ax_bar.set_title(title)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for x_pos, (combo, _count) in zip(x_positions, top_items):
        rows = [expert_to_row[expert] for expert in combo]
        ax_matrix.scatter([x_pos] * len(rows), rows, s=90, color="#2f6db2", zorder=3)
        if len(rows) > 1:
            ax_matrix.plot([x_pos, x_pos], [min(rows), max(rows)], color="#2f6db2", linewidth=2)

    ax_matrix.set_yticks(list(range(len(all_experts))))
    ax_matrix.set_yticklabels([f"Expert {expert}" for expert in all_experts])
    ax_matrix.set_xticks(x_positions)
    ax_matrix.set_xticklabels(labels, rotation=45, ha="right")
    ax_matrix.set_xlabel("Activated expert combination")
    ax_matrix.grid(axis="y", linestyle="--", alpha=0.3)
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_run_comparison_upset(intersection_records, path, title, max_combinations=18):
    """
    Save an upset-style plot across multiple runs.
    Each record should include:
    - run_label
    - combination
    - count
    - mean_layer_iou
    """

    if not intersection_records:
        raise ValueError("No intersection records were provided.")

    top_items = sorted(
        intersection_records,
        key=lambda item: (
            -item["count"],
            -item["mean_layer_iou"],
            item["run_label"],
            tuple(item["combination"]),
        ),
    )[:max_combinations]

    all_experts = sorted(
        {
            int(expert)
            for item in top_items
            for expert in item["combination"]
        }
    )
    expert_to_row = {expert: idx for idx, expert in enumerate(all_experts)}

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.05)
    ax_bar = fig.add_subplot(grid[0])
    ax_matrix = fig.add_subplot(grid[1], sharex=ax_bar)

    x_positions = list(range(len(top_items)))
    counts = [item["count"] for item in top_items]
    colors = [plt.cm.viridis(item["mean_layer_iou"]) for item in top_items]
    labels = [
        f'{item["run_label"]}\n' + "{" + ",".join(str(expert) for expert in item["combination"]) + "}\n"
        + f'Layer-IoU={item["mean_layer_iou"]:.2f}'
        for item in top_items
    ]

    ax_bar.bar(x_positions, counts, color=colors)
    ax_bar.set_ylabel("Examples")
    ax_bar.set_title(title)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for x_pos, item in zip(x_positions, top_items):
        rows = [expert_to_row[int(expert)] for expert in item["combination"]]
        ax_matrix.scatter([x_pos] * len(rows), rows, s=90, color=colors[x_pos], zorder=3)
        if len(rows) > 1:
            ax_matrix.plot([x_pos, x_pos], [min(rows), max(rows)], color=colors[x_pos], linewidth=2)

    ax_matrix.set_yticks(list(range(len(all_experts))))
    ax_matrix.set_yticklabels([f"Expert {expert}" for expert in all_experts])
    ax_matrix.set_xticks(x_positions)
    ax_matrix.set_xticklabels(labels, rotation=45, ha="right")
    ax_matrix.set_xlabel("Run-specific activated expert combinations with mean layer-set IoU")
    ax_matrix.grid(axis="y", linestyle="--", alpha=0.3)
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
