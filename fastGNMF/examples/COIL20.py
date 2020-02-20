"""
COIL20 dataset (source: http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)

Ensure a directory named "COIL20" containing the images is available under the examples directory
  with format obj[i]__[j].png where i = [1..n] and j = [0..k-1], n = # objects, k = # images/object

For example, with n = 2 and k = 3
- examples
  - COIL20
    - obj1__0.png
    - obj1__1.png
    - obj1__2.png
    - obj2__0.png
    - obj2__1.png
    - obj2__2.png
  - COIL20.py <this file>
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath
from PIL import Image
from PIL.ImageOps import expand
import seaborn as sns
from sklearn.manifold import TSNE

import sys
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import argparse

from fastGNMF import Gnmf, Nmf

# easy to differentiate colors
DISTINCT_COLS = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                 "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
                 "#469990", "#e6beff", "#9A6324", "#000000", "#800000",
                 "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"]

def read_dataset(rank=20, image_num=72, seed=None):
    """
    Read COIL20 dataset, resize to 32x32
    Parameters
    - rank     : the number of objects (k)
    - image_num: the number of images per object
    Returns
    - X          : [(32 x 32) x (rank x image_num)] matrix containing the images
    - groundtruth: an array with length image_num containing integers [0..rank-1],
                   separating the images into those categories
    """
    coil20_dir = join(dirname(abspath(__file__)), "COIL20")
    coil20_len = 20
    coil20_obj_len = 72
    img_size = 32

    if seed: np.random.seed(seed)
    # pick categories (total = rank)
    selected_cats = np.random.choice(coil20_len, rank, replace=False)

    # initiate X
    X = np.zeros((img_size * img_size, rank * image_num))
    # populate X
    for obj_i, cat in enumerate(selected_cats):
        for obj_img_j in range(image_num):
            # open and resize the image
            img = Image.open(join(coil20_dir, "obj%d__%d.png" % (obj_i + 1, obj_img_j))).resize((img_size, img_size))
            img_n = obj_i * image_num + obj_img_j
            X[:, img_n] = np.asarray(img).flatten()
    groundtruth = np.arange(rank * image_num)
    # shuffle images randomly
    np.random.shuffle(groundtruth)
    X = X[:,groundtruth]
    groundtruth = (groundtruth / image_num).astype(int)
    return(X, groundtruth)

def plot_basis(U, ncol, nrow, size=32):
    """
    Plots all basis images
    - U     : the basis matrix
    - ncol  : # cols in the canvas
    - nrow  : # rows in the canvas
    - size  : the height/width of each basis image (assuming it's a square)
    """
    # sns.heatmap(U)
    # plt.show()
    # return
    plt.set_cmap("gray")
    canvas = Image.new("L", (ncol * size + ncol+1, nrow * size + nrow+1)) # (w, h)
    for i in range(nrow):
        for j in range(ncol):
            basis = U[:, i * ncol + j].reshape((size, size))
            basis = basis / basis.max() * 255
            img = expand(Image.fromarray(basis), border=1, fill=255)
            canvas.paste(img.copy(), (j * size + j, i * size + i))
    plt.imshow(canvas)
    plt.show()

def visualize_tsne(V, rank, groundtruth, plot_title, plot_file, tsne_perplexity=2, seed=None):
    """
    Output V visualization using T-SNE
    - V              : the latent feature factor produced by the factorization
    - rank           : the number of clusters
    - groundtruth    : a list of integers represent different clusters
    - plot_title     : the plot title
    - plot_file      : the output file path
    - tsne_perplexity: the perplexity parameter for T-SNE
    - seed           : seed if necessary
    """
    # obtain clusters from V
    if seed: np.random.seed(seed)
    clusters = np.argmax(V, axis=0)
    V_embedded = TSNE(n_components=2, perplexity=tsne_perplexity).fit_transform(V.T)
    # visualize the tsne
    sns.scatterplot(V_embedded[:,0], V_embedded[:,1], hue=groundtruth,
                    palette=DISTINCT_COLS[:max(groundtruth)+1], legend=False,
                    linewidth=0
                    ).set_title(plot_title)
    plt.savefig(plot_file)
    plt.clf()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run GNMF on COIL20 dataset")
    parser.add_argument("-k", "--rank", type=int, default=20, help="The number of categories/rank used in GNMF")
    parser.add_argument("-n", "--imagenum", type=int, default=72, help="The number of images in each category")
    parser.add_argument("-p", "--pneighbor", type=int, default=5, help="The number of nearest neighbors to be considered")
    parser.add_argument("-l", "--lmbda", type=float, default=10, help="The lambda used for the regularizer")
    parser.add_argument("-i", "--iters", type=int, default=100, help="The # iterations to be run")
    parser.add_argument("-mt", "--method", type=str, default="euclidean", help="The update method: divergence or euclidean")
    parser.add_argument("-tp", "--perplexity", type=int, default=5, help="The t-sne perplexity")
    parser.add_argument("-s", "--seed", type=int, default=12345, help="Seed")
    input = parser.parse_args()

    best_obj_val = float("Inf")
    best_U = None
    best_V = None
    best_seed = None

    # Repeat the GNMF 20x and keep the one with the lowest objective function value
    for i in range(20):
        input.seed = np.random.randint(0, 1000)
        X, groundtruth = read_dataset(rank=input.rank, image_num=input.imagenum, seed=input.seed)
        gnmf = Gnmf(X=X, rank=input.rank, p=input.pneighbor, lmbda=input.lmbda, method=input.method)
        U, V, obj_values = gnmf.factorize(n_iter=input.iters, return_obj_values=True, seed=input.seed)

        # nmf = Nmf(rank=input.rank, method=input.method)
        # U, V, obj_values = nmf.factorize(X, n_iter=input.iters, return_obj_values=True, seed=input.seed)
        obj_val = obj_values[-1]

        if obj_val < best_obj_val:
            best_obj_val = obj_val
            best_U = U
            best_V = V
            best_seed = input.seed

    visualize_tsne(V, input.rank, groundtruth,
                    plot_title="pneighbor: %d, lmbda: %.2f, iterations: %d" % (input.pneighbor, input.lmbda, input.iters),
                    plot_file="gnmf_k-%d_n-%d_p-%d_l-%.2f_iter-%d.png" % (input.rank, input.imagenum, input.pneighbor, input.lmbda, input.iters),
                    tsne_perplexity=input.perplexity, seed=input.seed)
