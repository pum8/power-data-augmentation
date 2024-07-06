# Project Title: Data Synthesis using CTGAN,TVAE, and TabDDPM

## Overview
This project involves using the Conditional Tabular Generative Adversarial Network (CTGAN), Tabular Variational Autoencoder (TVAE), and Tabular Denoising diffusion probabilistic(TabDDPM) to synthesize realistic data based on the given dataset. The project includes implementations of the models, the generated figures, and the saved models for further data generation tasks.

## Project Structure

### CTGAN Synthesizer
- **CTGAN.py**: Script implementing the CTGAN model.
- **graph_generate.py**: Script used for graph and data generation.
- **model_CTGAN.pkl**: Saved CTGAN model, ready for data generation tasks.
- **perf_events_pwr.xlsx**: Original dataset file.
- **ctganfigure/**: Folder containing figures related to CTGAN data.

### TVAE Synthesizer
- **TVAE.py**: Script implementing the TVAE model.
- **graph_generate.py**: Script used for graph and data generation.
- **model_TVAE.pkl**: Saved TVAE model, ready for data generation tasks.
- **perf_events_pwr.xlsx**: Original dataset file.
- **figure/**: Folder containing figures related to TVAE data.

### TabDDPM Synthesizer
- **Taddpm.py**: Script implementing the diffusion model.
- **graph_generate.py**: Script used for graph and data generation.
- **ddpm_model.pkl**: Saved Tabddpm model, ready for data generation tasks.
- **perf_events_pwr.xlsx**: Original dataset file.
- **figure/**: Folder containing figures related to TVAE data.
- **hyperparameter.py**: hyperparameter tuning using the optuna framework.

## Version Requirements
- **Python**: 3.10.10
- **SDV**: 1.12.1

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/pum8/Green-computing-project.git
   cd project-name
