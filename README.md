# SL-MTGNN: Macroscopic Graph Topology-Based Multimodal Learning Framework for Robust Molecular Toxicity Prediction

SL-MTGNN is a multimodal deep learning framework that effectively predicts complex toxicological endpoints (e.g., drug-induced bone toxicity) by constructing a dynamic macroscopic drug-drug similarity graph to compensate for the lack of high-resolution biological features.

---

## 💡 Overview
State-of-the-art AI models often face a "feature starvation" dilemma when addressing stealthy and slow-developing endpoints like drug-induced bone toxicity. SL-MTGNN overcomes this by:
* Deeply integrating 4 heterogeneous chemical modalities (physicochemical descriptors, molecular fingerprints, contextual sequences, and microscopic molecular graphs).
* Constructing a macroscopic **Drug-Drug Similarity Graph**
* Introducing a topology-aware edge loss to explicitly model latent topological correlations, preventing representation collapse.

---

## quick start
⚙️ Installation
We highly recommend using Miniconda to manage your Python environment. Before quick start, you need to download the code related to MolFormer(https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct).
1. Clone the repository:
```
git clone https://github.com/hhuang-hub/MacroTox.git
cd MacroTox
```
2. Create and activate a conda environment:
```
conda env create -f environment.yml
conda activate macrotox
```

📊 Data Preparation
Place your raw dataset (in CSV format) into the data/ directory. The CSV file must contain at least two essential columns:
- smiles: The SMILES strings representing the molecular structures.
- [target_endpoint]: The toxicity label (e.g., ACEA_T47D_80hr_Negative for specific assays, or binary 0/1 labels for toxicity).
generate graph_feature
```
python graph_process.py --config config.yaml
```

📈 Usage
Once your environment is set up and the data is ready, you can start the feature extraction and model training pipeline with a single command:
```
python train.py --config config.yaml
```
After training, evaluate the model's performance on test set:
```
python test.py --config config.yaml
```

## 🚀 direct usage
Prepare the input data as a .csv file or a single SMILES string. Users can perform predictions via the command line as follows:

```
python predict.py <input_data_or_SMILES> <output_path> <graph_root>
```

