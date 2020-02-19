import numpy as np
import time

class Gnmf(object):
    """
    Graph regularized NMF

    Note: the divergence update method is not yet finalized
    """
    def __init__(self, X, rank=10, p=3, W=None, lmbda=100, method="euclidean",
                 knn_index_type="IndexFlatL2", knn_index_args=None):
        """
        - X     : the original matrix
        - rank  : NMF rank
        - p     : # closest neighbors to be taken into account in the weight matrix
        - W     : the weight matrix - must be symmetric; p will be ignored if W is provided
        - method: "euclidean" or "divergence"
        - knn_index_type: faiss index type to compute k-nearest neighbors in matrix generation
        - knn_index_args: faiss index arguments; if None, by default the argument for IndexFlatL2 is used
        Check https://github.com/facebookresearch/faiss/wiki/Faiss-indexes for further details on faiss index
        """
        self.X = X
        self.rank = rank
        self.method = method
        self.lmbda = lmbda

        # weight matrix variables
        self.W = W
        self.p = p
        self.knn_index_type = knn_index_type
        self.knn_index_args = knn_index_args
        if self.W is None:
            self.W = self.calc_weights_matrix()
        elif not self.is_matrix_symmetric(self.W):
            raise ValueError("The provided weight matrix should be symmetric")

        # calc the Laplacian matrix L
        self.D = np.diag(np.sum(self.W, axis=0))
        self.L = self.D - self.W

    def init_rand_matrix(self, nrow, ncol, seed=None):
        """
        Initialize matrix (random) given # rows and # cols
        """
        if not seed:
            seed = np.random.randint(1000)
        np.random.seed(seed)
        return(np.random.dirichlet(np.ones(nrow), size=ncol).T)

    def calc_weights_matrix(self):
        """
        Generate weights matrix by faiss (facebook AI similarity search) + dot-product weighting
        """
        import faiss

        print("Generating weight matrix using faiss...")
        time_cp1 = time.time()
        X = self.X.astype(np.float32)
        n, m = X.shape
        xb = np.ascontiguousarray(X.T) # database ~ rows of vectors
        xq = np.ascontiguousarray(X.T) # query vectors
        W = np.zeros((m, m))

        p = self.p
        if not self.knn_index_args:
            self.knn_index_args = (n,)
        # build the index
        index = getattr(faiss, self.knn_index_type)(*self.knn_index_args)
        index.add(xb)                  # add vectors to the index
        _, I = index.search(xq, p+1)   # the first col would be the vector itself

        for i in range(m):
            for j in range(p):
                neighbor = I[i][j+1]
                # compute dot-product weighting
                W[i][neighbor] = W[neighbor][i] = np.dot(X[:,i], X[:,neighbor])
        time_cp2 = time.time()
        print("total time: ", time_cp2 - time_cp1)
        return(W)

    def update_euclidean(self, U, V):
        """
        Update U & V using multiplicative euclidean approach
        """
        X = self.X
        W = self.W
        lmbda = self.lmbda
        L = self.L
        D = self.D
        # update V
        V = V * np.divide(U.T @ X + lmbda * (V @ W), U.T @ U @ V + lmbda * (V @ D))
        # update U
        U = U * np.divide(X @ V.T, U @ V @ V.T)
        # calc objective func
        R = X - (U @ V)
        obj_val = np.sum(R * R) + lmbda * np.trace(V @ L @ V.T)
        return(U, V, obj_val)

    def np_pos(self, np_ar, add_eps=False):
        """
        Ensure all values in a numpy array > 0
        """
        eps = np.finfo(np_ar.dtype).eps
        if add_eps:
            return(np_ar + eps)
        np_ar[np_ar == 0] = eps
        return(np_ar)

    def update_divergence(self, U, V):
        """
        Update U & V using multiplicative divergence approach
        """
        X = self.X
        lmbda = self.lmbda
        L = self.L
        n, m = X.shape
        k, _ = V.shape

        # update V
        #TODO*: improve using iterative algorithm CG
        V = V * (U.T @ np.divide(X, U @ V))
        U_row_sum = np.sum(U, axis=0).reshape((k, 1))
        for i in range(k):
            V[i] = V[i] @ np.linalg.pinv(U_row_sum[i] * np.identity(m) + lmbda * L)
        # update U
        V_col_sum = np.sum(V, axis=1).reshape((1, k))
        U = U * np.divide(np.divide(X, U @ V) @ V.T, V_col_sum)
        # calc obj_val
        X_temp = U @ V
        obj_val = np.sum(X * np.log(
            self.np_pos(np.divide(self.np_pos(X), self.np_pos(X_temp)), add_eps=True)
            ) - X + X_temp)
        return(U, V, obj_val)

    def is_matrix_symmetric(self, M, rtol=1e-05, atol=1e-08):
        """
        Check if the given matrix M is symmetric
        """
        return(np.allclose(M, M.T, rtol=rtol, atol=atol))

    def factorize(self, n_iter=100, return_obj_values=False, seed=None):
        """
        Factorize matrix X into W and H given rank using multiplicative method
        params:
        - n_iter           : the number of iterations
        - return_obj_values: enable returning the list of produced objective function values as the third tuple element
        Returns a tuple (U, V) or (U, V, obj_values if return_obj_values is True)
        """
        X = self.X
        n, m = X.shape
        rank = self.rank
        method = self.method
        W = self.W

        # initialize U & V
        U = self.init_rand_matrix(n, rank, seed)
        V = self.init_rand_matrix(rank, m, seed)

        print("running GNMF given matrix %dx%d, rank %d, %d neighbors, lambda %d, %d iterations" % (n, m, rank, self.p, self.lmbda, n_iter))
        time_cp1 = time.time()
        obj_vals = [] # a list of the produced objective function values
        curr_obj_val = float("inf")
        for iter in range(n_iter):
            if iter % 20 == 0:
                print("Completed %d iterations..." % iter)

            U, V, obj_val = (self.update_euclidean(U, V)
                            if self.method == "euclidean"
                            else self.update_divergence(U, V))
            print("Objective function value is %.2f" % obj_val)

            # check if the objective function value is decreasing
            if curr_obj_val < obj_val:
                print("The objective function value is not decreasing!! :'(")
                break
            curr_obj_val = obj_val
            obj_vals.append(obj_val)

        time_cp2 = time.time()
        print("total duration: %d; avg duration/iter: %.2f" % (time_cp2 - time_cp1, (time_cp2 - time_cp1) / n_iter))

        # set the euclidean length of each col vec in U = 1
        sum_col_U = np.sqrt(np.sum(U**2, axis=0))
        V = V * sum_col_U.reshape((rank, 1))
        U = U / sum_col_U

        if return_obj_values:
            return(U, V, obj_vals)
        return(U, V)

    def get_weight_matrix(self):
        """
        Retrieve the weight matrix W
        """
        return self.W
