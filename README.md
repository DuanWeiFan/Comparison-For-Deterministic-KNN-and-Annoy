# Comparison-For-Deterministic-KNN-and-Annoy
This is the implementation of final project for CU Boulder Course 5454

# Objective
The objective of this project is to compare the accuracy and complexity between Deterministic KNN and Annoy Algorithm. Note that this implementation of Annoy Algorithm is trying to replicate the general idea of Annoy https://github.com/spotify/annoy, yet it misses some advanced features.

# Prerequisite
1. Run `pip install -r requirements.txt`
2. Download MNIST train dataset from `https://www.kaggle.com/c/digit-recognizer/data`
3. In annoy.py, set MNIST_PATH to the downloaded MNIST train dataset csv file

# Run
## Run Deterministic KNN
`python deterministic_knn.py`

## Run Annoy KNN
`python annoy.py`

