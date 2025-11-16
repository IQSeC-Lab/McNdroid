# M3-Analysis: Multimodal Malware Modeling

This repository contains a GPU-ready research notebook that synthesizes multimodal malware telemetry and trains a transformer-based classifier inspired by modern Vision Transformers (ViT) and large language models (LLMs). The goal is to provide a realistic sandbox for experimenting with modality fusion, interpretability, and reasoning for malware analysis workflows.

## Project Highlights
- **Four complementary modalities**: static metadata fields, dynamic execution traces, control-flow/API graphs, and network telemetry. Each modality is produced by a distinct stochastic process to mimic real-world variance.
- **Latent threat generator**: labels are not random—they stem from modality-weighted risk factors (e.g., beacon-like network spikes), making the classification task meaningful.
- **ViT-style fusion model**: dense vectors are chunked into patch tokens, fused with a deep transformer stack, and paired with dedicated CLS/REASON tokens for classification and interpretability.
- **Reasoning outputs**: the model surfaces modality attribution weights so analysts can see why a sample is considered malicious.
- **Reproducible pipeline**: the notebook builds the dataset, trains the model, evaluates ROC/F1, and runs entirely inside the provided `multimodal-demo` conda environment.

## Repository Layout
```
M3-Analysis/
├── README.md                # Project overview (this file)
├── requirements.txt         # Python dependencies for the notebook
└── multimodal_malware_demo.ipynb  # Main GPU-ready research notebook
```

## Requirements
Core libraries (see `requirements.txt` for installable versions):
- PyTorch 2.6.0 (CUDA 12.4 build recommended)
- TorchVision / TorchAudio 0.21.0
- NumPy 2.2+, scikit-learn 1.5+
- tqdm, JupyterLab, nbconvert, ipykernel

### Suggested Environment Setup
```bash
# 1. Create the conda environment
conda create -n multimodal-demo python=3.10 -y
conda activate multimodal-demo

# 2. Install PyTorch with CUDA 12.4 wheels (adjust if you need CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install the rest of the stack
pip install -r requirements.txt

# 4. (Optional) register the kernel for VS Code / Jupyter
python -m ipykernel install --user --name multimodal-demo --display-name "Python (multimodal-demo)"
```

## Running the Notebook
You can interactively run the notebook in VS Code or execute it headlessly:

```bash
cd /home/msrahman3/Research_Projects/Multi-Modal-Demo
conda activate multimodal-demo
jupyter nbconvert --to notebook --inplace --execute M3-Analysis/multimodal_malware_demo.ipynb \
  --ExecutePreprocessor.kernel_name=multimodal-demo --ExecutePreprocessor.timeout=1200
```

The final cell prints classification metrics (Accuracy/Precision/Recall/F1/AUC) and per-modality reasoning weights derived from the REASON token. The notebook also saves the best-performing checkpoint to `best_multimodal_model.pt` for reuse across sessions.

## Model Notes & Future Work
- **Vision Transformer inspiration**: modality vectors are split into fixed-size "patches" before entering a 6-layer transformer encoder with gelu activations, rotary-style positional mixing, and learnable gating parameters.
- **LLM-style reasoning**: a dedicated REASON token and context projections estimate which modality contributed most to the decision, forming a lightweight rationale head.
- **Possible extensions**:
  1. Swap synthetic generators with real malware telemetry pipelines (IDA static dumps, ETW traces, C2 flow data).
  2. Add modality-specific encoders (CNN for image-like static artifacts, sequence models for API call traces).
  3. Introduce adversarial or hard-negative data synthesis to stress generalization.
  4. Log attention maps / rationale scores per sample for analyst-facing reports.

## License
This repository currently has no explicit license. Please add one before distributing outside your organization.
