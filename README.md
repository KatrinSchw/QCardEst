# QCardEst/QCardCorr

This repository is a research fork of the original **QCardEst/QCardCorr** implementation by Winker et al. The project implements hybrid quantum-classical machine learning models for cardinality estimation and correction in database query optimization.

**Original paper:**
> T. Winker et al., *QCardEst/QCardCorr: Quantum Cardinality Estimation and Correction*

This fork extends the original implementation with:
- Extended experimental evaluation on JOB-light and STATS benchmarks
- Comprehensive documentation of the hybrid quantum-classical model architecture
- Reproducible analysis scripts and visualization tools
- Detailed evaluation reports with performance comparisons

All original concepts and methods are credited to the authors.

## Installation

This project uses a virtual environment defined by 'environment.yml' to manage dependencies.

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

All extended documentation is provided as PDFs in the docs/ directory:
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

## License

This is a research fork. Please refer to the original repository for licensing information and cite the original paper when using this code.

## Citation

If you use this implementation, please cite:

```
T. Winker et al., "QCardEst/QCardCorr: Quantum Cardinality Estimation and Correction"
```
