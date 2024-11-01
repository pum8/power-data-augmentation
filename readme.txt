# Data Synthesis using CTGAN, and TabDDPM

## Overview
This project leverages the Conditional Tabular Generative Adversarial Network (CTGAN) and Denoising Diffusion Probabilistic Model (DDPM) to synthesize realistic data based on a given dataset. The repository includes implementations of these models, generated figures, and saved models for further data generation tasks.

## Project Structure

### CTGAN Synthesizer
- **CTGAN.py**: Script implementing the CTGAN model.
- **graph_generate.py**: Script used for graph and data generation.
- **model_CTGAN.pkl**: Saved CTGAN model, ready for data generation tasks.
- **perf_events_pwr.xlsx**: Original dataset file.
- **ctganfigure/**: Folder containing figures related to CTGAN data.


### DDPM Synthesizer
- **Taddpm.py**: Script implementing the DDPM model.
- **graph_generate.py**: Script used for graph and data generation.
- **ddpm_model.pkl**: Saved DDPM model, ready for data generation tasks.
- **perf_events_pwr.xlsx**: Original dataset file.
- **figure/**: Folder containing figures related to TVAE data.
- **hyperparameter.py**: Hyperparameter tuning using the Optuna framework.

## Version Requirements
- **Python**: 3.10.10
- **SDV**: 1.12.1

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/pum8/Green-computing-project.git
   cd Green-computing

##  Usage
You can use the saved models (model_CTGAN.pkl, ddpm_model.pkl) to generate new data based on the original dataset (perf_events_pwr.xlsx). The generated figures for each model are stored in their respective folders


