import seaborn as sns
import matplotlib.pyplot as plt



def plot_expert_heatmap(matrix):

    plt.figure(figsize=(8,4))

    sns.heatmap(matrix, cmap="viridis")

    plt.xlabel("Expert")
    plt.ylabel("Layer")

    plt.title("Layer × Expert Usage")

    plt.show()