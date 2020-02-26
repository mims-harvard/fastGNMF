# fastGNMF
Fast graph-regularized non-negative matrix factorization based on [faiss](https://github.com/facebookresearch/faiss) for finding p-nearest neighbors.

Current version: 0.1.0

## Installation

```
python setup.py install --user
```

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
