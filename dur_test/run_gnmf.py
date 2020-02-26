"""
Compare the duration of running fastGNMF vs vanilla GNMF given different matrix sizes
"""

import numpy as np
import os
from os.path import join, dirname, abspath, exists
import sys
import time
import argparse

import fastGNMF

rank = 50
p    = 8
result_csv_file = "results.csv"

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Generate X and W matrices for GNMF testing")
    parser.add_argument("-n", "--height", type=int, help="The height of the V matrix", required=True)
    parser.add_argument("-m", "--width", type=int, help="The width of the V matrix", required=True)
    parser.add_argument("-p", "--pneighbor", type=int, default=5, help="The number of nearest neighbors to be considered")
    parser.add_argument("-k", "--rank", type=int, default=20, help="The number of categories/rank used in GNMF")
    parser.add_argument("--vanilla", dest="vanilla", action="store_true", help="Run vanilla GNMF", default=False)
    input = parser.parse_args()
    print(input)

    n = input.height
    m = input.width
    disable_faiss = input.vanilla

    X = np.random.rand(input.height, input.width)
    time_cp1 = time.time()
    gnmf = fastGNMF.Gnmf(X=X, rank=rank, p=p, disable_faiss=disable_faiss)
    gnmf.factorize()
    time_cp2 = time.time()
    tot_dur = time_cp2 - time_cp1
    if not exists(result_csv_file):
        with open(result_csv_file, "w") as file:
            file.write("n,m,dur,vanilla\n")
    with open(result_csv_file, "a") as file:
        file.write("%d,%d,%.2f,%s" % (n, m, tot_dur, "yes" if disable_faiss else "no"))
