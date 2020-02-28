# fastGNMF
Fast graph-regularized non-negative matrix factorization based on [faiss](https://github.com/facebookresearch/faiss) for finding p-nearest neighbors.

Current version: 0.1.1

## Runtime

<img src="https://user-images.githubusercontent.com/7066351/75503577-f731d180-59a3-11ea-85e6-ae2e2c264404.png" width="500" />

(run on the same machine using randomly generated data matrix)

Other parameters used:

```
- lambda      = 0.5
- k           = 50
- p           = 8
- faiss index = IndexFlatL2
- # iterations= 100
```

## Installation

### 1) Build and install the module

```
python setup.py install --user
pip install -r requirements.txt
```

### 2) Install faiss library

Follow the steps [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).

## Quick Start

Running GNMF on the example COIL20 dataset.

```
import fastGNMF

# groundtruth ~ to obtain the cluster labels
X, groundtruth = fastGNMF.examples.COIL20.read_dataset(rank=10, image_num=5)

# initialize gnmf instance with rank=10 and p=5 for p-nearest neighbors
#  use default parameters for the rest (lambda = 0.5)
gnmf = fastGNMF.Gnmf(X=X, rank=10, p=4)
U, V = gnmf.factorize()

# output a t-sne image
fastGNMF.examples.COIL20.visualize_tsne(V, 10, groundtruth, "COIL20 test with rank=10; lambda=0.5; p=4", "test.png", tsne_perplexity=5)
```

The code above will output and save an image below.

![test_image](https://user-images.githubusercontent.com/7066351/74950403-0d042d00-53cd-11ea-870a-8a047bed09b5.png)

More detailed documentation can be found on the [Wiki](https://github.com/mims-harvard/fastGNMF/wiki) page.
