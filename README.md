# LWCC
This is the repository for Label-Wise base learner configuration.
## Requirements
- python 3.11 or later
- pyglove
- numpy
- pandas
- sklearn
- sklearnex
- scikit-multilearn

# Benchmarking
To run the benchmark use the following command:

``python -u ".\code\benchmarking_v1.py" SEED DATASET CLEAN_UP --prog_bar True``

- SEED is an integer used to set the random seed in the algorithm
- DATASET is the string for the dataset e.g. "image"
- CLEAN_UP is a boolean used to delete the cache of trained classifiers after the benchmark

# Results
The results are stored in the results dictionary. The dictionary name is determined by the dataset name and order. The results are saved in separate small CSV files to avoid overcomplicating parallel processing.