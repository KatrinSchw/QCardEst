# QCardEst

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

### Dataformat

The data files contain 6 columns seperated by ','

1. Column : Tablenames of the query seperated by ';'
2. Column : Selectivities for the tables seperated by ';'
3. Column : Cost calculated by PostgreSQL
4. Column : Cardinality prediction of PostgreSQL/MSCN
5. Column : Actual execution time
6. Column : True cardinality

### Settings
    
The important setting options are:

- data: The datafile which should be used 
    - jobSimple/job : JOB-light benchmark with PostgreSQL as classical base
    - jobSimple/mscnCosts : JOB-light benchmark with PostgreSQL as classical base
    - stats/statsCards6 : STATS benchmark with up to 6 tables in query
    - stats/stats4 : STATS benchmark with exactly 4 tables in query
    - stats/statCards : STATS benchmark with all queries (up to 7 tables query)
    
- valueType : select cost to predict and method of predicting
    - rows : Cardinality prediction
    - rowFactor : Cardinality correction
    - cost : Execution time prediction
    - costFactor : Execution time calculation from predicted cost

- numEpisodes : Number of episodes the optimizer runs
- loss : Classical post-processing layer
- prefix : Prefix for the result filename
- reps : Number of layers of the VQC 

## Additional Documentation and Analysis

This fork extends the original implementation with:
- Extended evaluation and analysis
- Reproducible plots and experiment scripts
- LaTeX documentation of the hybrid quantum-classical model and experiments

### Documentation
- `docs/01_preliminaries.pdf`: Background and related work
- `docs/02_hybrid_model.pdf`: Compact encoding and hybrid VQC model
- `docs/03_evaluation.pdf`: Experimental setup and results

### Analysis
- `analysis/`: Evaluation plots and post-processing scripts 

