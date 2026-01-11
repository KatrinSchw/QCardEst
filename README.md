# QCardEst / QCardCorr

This repository contains a research implementation and extended analysis of the
hybrid quantum–classical cardinality estimation models **QCardEst** and **QCardCorr**
originally proposed by Winker et al.

**Original paper:**
> T. Winker et al., *QCardEst/QCardCorr: Quantum Cardinality Estimation and Correction*

The codebase is based on the original implementation released by the authors (https://github.com/TobiasWinker/QCardEst) and has
been adapted and extended for research and evaluation purposes. Extensions in this
repository include:

- Additional experimental evaluation on JOB-light and STATS benchmarks
- Detailed analysis of classical post-processing layers
- Reproducible plotting and analysis scripts
- Structured LaTeX documentation of the model architecture and evaluation

All core concepts, model designs, and algorithms originate from the original authors.
This repository does not claim original authorship of the QCardEst or QCardCorr methods.

## Installation

This project uses a Conda environment defined in `environment.yml` to manage dependencies.

1. Install Anaconda
2. Create environment with `conda env create -f environment.yml`
3. Activate environment with `conda activate QCardEst`

## Execution

Run with 

```python runRegression.py``` 

The settings can be added in the file or passed as command line arguments e.g.

```python runRegression.py '{"reps":3}'```

For numEpiosdes=8000 and reps=16 this can take up to 2 days to finish.

### Documentation

The main written documentation is provided as compiled PDFs in the `docs/` directory:
- **`docs/explanation_hybrid_model.pdf`**: Technical explanation of the hybrid quantum-classical architecture, including compact encoding strategies and the variational quantum circuit design.

- **`docs/evaluation.pdf`**: Comprehensive evaluation report comparing QCardEst and QCardCorr on JOB-light and STATS benchmarks, analyzing the impact of different post-processing layers on estimation and correction accuracy.

### Code Documentation

Additional implementation-level documentation is provided in the analysis/ directory:
- **`analysis/COMPUTATION_GUIDE.pdf`**: Guide to key computation functions, entry points, and the difference between QCardEst and QCardCorr implementations.

- **`analysis/EXPERIMENTAL_SETUP.pdf`**: Documentation of experimental setup parameters, their locations in the codebase, and configuration options.

## Analysis and Results

The `analysis/` directory contains:

- **Plot generation scripts**: 
  - `make_plots_from_results.py`: Main plotting script
  - `create_job_light_figure.py`: JOB-light specific visualizations
  - `create_stats_figure.py`: STATS benchmark visualizations

- **Generated figures**: 
  - Training curves for estimation and correction tasks
  - Prediction quality metrics
  - Classical layer comparison analyses

## Usage and Attribution

**Copyright Notice:** The original code, algorithms, and model designs are copyright of Winker et al.
The original implementation does not specify an explicit software license, and therefore all rights
are reserved by the original authors.

This repository is intended for academic, research, and reproducibility purposes only.
All core algorithms, model designs, and methodological ideas originate from the original authors.
This repository provides extensions in the form of evaluation, analysis, and documentation,
and does not claim ownership of the original implementation or its components.

If you intend to reuse or redistribute the original code beyond academic study,
please consult the original authors.

## Citation

If you use this repository or build upon the original models, please cite the
original work:
```
T. Winker et al., "QCardEst/QCardCorr: Quantum Cardinality Estimation and Correction"
```
