import seaborn as sns
import matplotlib.pyplot as plt



def expert_heatmap(matrix, title="Expert usage"):

    plt.figure(figsize=(6,4))

    sns.heatmap(matrix, cmap="viridis")

    plt.title(title)

    plt.xlabel("Expert")
    plt.ylabel("Layer")

    plt.show()