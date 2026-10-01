# Cora Citation Network Paper Classification

Classify research papers into one of **seven topics** using both their word features and the citation links between them. The project compares a classic **Random Forest baseline** against a **two-layer Graph Convolutional Network (GCN)**, and is deployed as two live web apps.

## Live Demos

| Deployment | Link |
|---|---|
| Render | https://cora-project-kqgk.onrender.com/ |
| Streamlit | https://coraprojectstr.streamlit.app/ |

> Free-tier hosting can sleep when idle, so the first load may take a little while.

## Overview

The goal is to answer one question: **does citation structure help predict a paper's topic?**

- **Baseline:** each paper is classified on its own, using only its features (plus degree and log-degree as light graph context).
- **GCN:** each paper's representation is also shaped by the papers it is connected to (message passing across the graph).

## Dataset

[Cora](https://paperswithcode.com/dataset/cora), loaded through PyTorch Geometric's `Planetoid` wrapper.

| Item | Meaning | Value |
|---|---|---|
| Nodes | Research papers | 2,708 |
| Edges | Citations | 10,556 (directed edge count as stored by PyG) |
| Node features | Binary word-presence indicators | 1,433 |
| Classes | Paper topics | 7 |
| Split | Standard Planetoid masks | 140 train / 500 val / 1,000 test |

## Models

### 1. Baseline: TF-IDF reweighting + Random Forest
- Binary word indicators are reweighted with `TfidfTransformer`.
- `RandomForestClassifier` with 100 trees, `max_features="sqrt"`, `class_weight="balanced"`.
- Purpose: measure how well papers can be classified **without** message passing.

### 2. Two-layer GCN
- `GCNConv(1433 → 32)` → ReLU → Dropout (0.5) → `GCNConv(32 → 7)`
- Adam optimizer, learning rate `0.01`, up to 200 epochs
- Early stopping with patience of 20 epochs on validation accuracy; the best checkpoint is restored for testing

Both models are evaluated on **accuracy** and **macro-F1** (macro-F1 weights every topic equally, which helps when class sizes differ).

## Results

Fill in from your notebook's final `results` table:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest | _your value_ | _your value_ |
| GCN | _your value_ | _your value_ |

Exact numbers can vary slightly with hardware, package versions, and random seed.

## Model Export

The trained GCN is exported to **ONNX** (`simple_gcn_cora.onnx`, opset 18) with dynamic axes for the number of nodes and edges, so it can be served outside PyTorch.

## Tech Stack

- Python, PyTorch, PyTorch Geometric
- scikit-learn, pandas, NumPy
- NetworkX, Matplotlib, Seaborn
- ONNX
- Streamlit and Render for deployment

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torch-geometric scikit-learn pandas matplotlib seaborn networkx onnxscript
```

Then open and run the notebook:

```bash
jupyter notebook cora_citation_network_classification.ipynb
```

The Cora dataset downloads automatically into `data/Planetoid` on first run.

## Project Structure

Adjust to match your repository:

```
.
├── cora_citation_network_classification.ipynb   # EDA, baseline, GCN, evaluation, ONNX export
├── simple_gcn_cora.onnx                         # exported GCN model
├── README.md
└── ...                                          # app files for Render / Streamlit
```

## Limitations and Next Steps

A GCN outperforming the baseline suggests citation structure carries useful signal, but it does not prove citations *cause* topic membership. The graph may contain homophily, dataset artifacts, or leakage through the standard benchmark split.

Ideas to strengthen the project:
- Repeat on **CiteSeer** and **PubMed**
- Report mean ± standard deviation over several random seeds
- Test a model that uses only graph features
- Try other architectures such as GAT or GraphSAGE

## References

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (2017)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): `Planetoid` and `GCNConv`

## License

Add your preferred license here (for example, MIT).
