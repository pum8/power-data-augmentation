# Project Title: Data Synthesis using CTGAN and TVAE

## Overview
This project involves the use of Conditional Tabular Generative Adversarial Network (CTGAN) and Tabular Variational Autoencoder (TVAE) to synthesize realistic data based on the given dataset. The project includes implementations of both models, the generated figures, and the saved models for further data generation tasks.

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

## Version Requirements
- **Python**: 3.10.10
- **SDV**: 1.12.1

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repository/project-name.git
   cd project-name
