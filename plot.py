import json

import matplotlib.pyplot as plt

results = {
    "scirepeval_results_bge-large-en-v1.5.json": 300,
    "scirepeval_results_bge-m3.json": 567,
    "scirepeval_results_embeddinggemma-300m.json": 300,
    "scirepeval_results_mxbai-embed-large-v1.json": 335,
    "scirepeval_results_nomic-embed-text-v1.5.json": 137,
    "scirepeval_results_Qwen3-Embedding-0.6B.json": 600,
    "scirepeval_results_specter2-base.json": 110,
    "scirepeval_results_specter2-base-adapters.json": 111,
}


def name_to_ndcg(file: str) -> float:
    with open(file, "r") as f:
        result = json.load(f)
    ndcg = result.get("Search").get("ndcg")
    return ndcg


model_names = [
    name.split("_")[-1].replace(".json", "") for name in list(results.keys())
]
model_parameters = [parameter * 10 for parameter in list(results.values())]
model_results = [name_to_ndcg(file) for file in list(results.keys())]

fig = plt.figure(figsize=(20, 10), dpi=150)
plt.grid() # color="0.95"

plt.scatter(
    model_names,
    model_results,
    s=model_parameters,
    c=model_parameters,
    cmap="jet",
    # alpha=0.95,
)
plt.xlabel("Model Name")
plt.ylabel("nDCG")
plt.title("Model Performance on SciRepEval (Bubble Size = Parameters)")

plt.savefig("bubble_plot_scirepeval.png")
