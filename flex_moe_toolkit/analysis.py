import pandas as pd
import numpy as np



def compute_expert_usage(log_path):

    df = pd.read_json(log_path, lines=True)

    counts = {}

    for probs in df["probs"]:

        experts = np.argsort(probs)[-5:]   # top-k

        for e in experts:

            counts[e] = counts.get(e, 0) + 1

    return pd.Series(counts).sort_index()


def layer_expert_matrix(log_path):

    df = pd.read_json(log_path, lines=True)

    num_experts = len(df.iloc[0]["probs"])
    num_layers = df["layer"].max() + 1

    matrix = np.zeros((num_layers, num_experts))

    for _, row in df.iterrows():

        probs = row["probs"]

        topk = np.argsort(probs)[-5:]

        for e in topk:
            matrix[row["layer"], e] += 1

    return matrix