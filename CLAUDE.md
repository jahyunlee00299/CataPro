# CataPro - Claude Code Documentation

## Project Overview

CataPro is a machine learning-based enzyme kinetic parameter prediction tool that predicts:
- Turnover number (kcat)
- Michaelis constant (Km)
- Catalytic efficiency (kcat/Km)

The project leverages protein language models (ProtT5), small molecule language models (MolT5), and molecular fingerprints to achieve state-of-the-art prediction performance.

## Project Structure

```
CataPro/
├── inference/              # Inference scripts and models
│   ├── predict.py         # Main prediction script
│   ├── model.py           # Model architecture
│   ├── act_model.py       # Activation model components
│   └── utils.py           # Utility functions
├── training/              # Training scripts for each parameter
│   ├── kcat/             # kcat training
│   ├── Km/               # Km training
│   └── kcat_over_Km/     # kcat/Km training
├── models/               # Pre-trained model weights directory
│   ├── prot_t5_xl_uniref50/      # Protein language model
│   └── molt5-base-smiles2caption/ # Molecule language model
├── samples/              # Sample input files
└── datasets/            # Training datasets
```

## Key Components

### 1. Model Architecture
- **Protein Encoder**: Uses ProtT5 (prot_t5_xl_uniref50) for enzyme sequence feature extraction
- **Molecule Encoder**: Uses MolT5 for substrate SMILES feature extraction
- **Prediction Head**: Combines protein and molecule features for kinetic parameter prediction

### 2. Data Format
Input data should be in CSV format with columns:
- `Enzyme_id`: Unique enzyme identifier
- `type`: wild-type or mutant
- `sequence`: Amino acid sequence
- `smiles`: Substrate SMILES notation

### 3. Training Approach
- Dataset clustered by 0.4 protein sequence similarity
- 10-fold cross-validation
- Separate models for kcat, Km, and kcat/Km

## Working with CataPro

### Running Predictions

```bash
cd inference
python predict.py \
    -inp_fpath samples/sample_inp.csv \
    -model_dpath models \
    -batch_size 64 \
    -device cuda:0 \
    -out_fpath catapro_prediction.csv
```

Or simply:
```bash
cd inference
bash run_catapro.sh
```

### Model Modifications

If you need to modify the model architecture:
1. **Inference model**: Edit `inference/model.py`
2. **Training models**: Edit corresponding files in `training/{kcat,Km,kcat_over_Km}/model.py`
3. **Activation functions**: Modify `inference/act_model.py`

### Adding New Features

To add new substrate or protein features:
1. Update feature extraction in `inference/utils.py`
2. Modify model input dimensions in `model.py`
3. Retrain models using training scripts

## Dependencies

Key requirements:
- PyTorch >= 1.13.0
- Transformers (for language models)
- RDKit (for molecular fingerprints)
- NumPy, Pandas (data processing)

## Common Tasks

### 1. Testing with New Data
Place your CSV file in the `samples/` directory and run predict.py with the appropriate path.

### 2. Retraining Models
Navigate to `training/{parameter}/` and run the training script with your dataset.

### 3. Analyzing Predictions
Use the data analysis utilities in the inference directory to visualize and compare predictions.

## Notes for Development

- The project uses pre-trained language models that must be downloaded separately
- Model weights are stored in the `models/` directory (not tracked in git due to size)
- GPU is recommended for inference (use `device cuda:0`)
- Batch size can be adjusted based on available GPU memory

## Contact

For questions about the implementation:
- Original Author: Zechen Wang, PhD (wangzch97@gmail.com)
- Institution: Shandong University

## Future Improvements

Potential areas for enhancement:
- [ ] Support for additional kinetic parameters
- [ ] Integration with protein structure prediction tools
- [ ] Web API for predictions
- [ ] Ensemble model approaches
- [ ] Support for more substrate representations
