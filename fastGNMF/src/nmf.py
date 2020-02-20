import numpy as np

class Nmf(object):
    """
    The basic non-negative matrix factorization (Lee & Seung, 1999 & 2001)
    """
    def __init__(self, rank=20, method="euclidean"):
        self.method = method
        self.rank = rank

    def init_rand_matrix(self, nrow, ncol, seed=None):
        """
        Initialize matrix (random) given # rows and # cols
        """
        if not seed:
            seed = np.random.randint(1000)
        np.random.seed(seed)
        return(np.random.dirichlet(np.ones(nrow), size=ncol).T)

    def update_euclidean(self, W, H, V):
        # update H
        H = H * np.divide(W.T @ V, W.T @ W @ H)
        # update W
        W = W * np.divide(V @ H.T, W @ H @ H.T)
        # calc objective func
        R = V - (W @ H)
        D = np.sum(R * R)
        return(W, H, D)

    def np_pos(self, np_ar, add_eps=False):
        """Ensures all values in a numpy array > 0"""
        eps = np.finfo(np_ar.dtype).eps
        if add_eps:
            return(np_ar + eps)
        np_ar[np_ar == 0] = eps
        return(np_ar)

    def update_divergence(self, W, H, V):
        n, k = W.shape
        # update H
        W_row_sum = np.sum(W, axis=0).reshape((k, 1))
        H = H * np.divide(W.T @ np.divide(V, W @ H), W_row_sum)
        # update W
        H_col_sum = np.sum(H, axis=1).reshape((1, k))
        W = W * np.divide(np.divide(V, W @ H) @ H.T, H_col_sum)
        # calc objective func
        V_temp = W @ H
        # obj_val = np.sum(V * np.log(np.divide(V, V_temp)) - V + V_temp)
        obj_val = np.sum(V * np.log(
        self.np_pos(np.divide(self.np_pos(V), self.np_pos(V_temp)), add_eps=True)
        ) - V + V_temp)

        return(W, H, obj_val)

    def factorize(self, V, n_iter=100, return_obj_values=False, seed=None):
        """
        Factorizes matrix V into W and H given rank using multiplicative method
        method options: ["euclidean", "divergence"]
        """
        n, m = V.shape
        rank = self.rank
        W = self.init_rand_matrix(n, rank, seed)
        H = self.init_rand_matrix(rank, m, seed)
        obj_vals = [] # a list of the produced objective function values
        for iter in range(n_iter):
            if self.method == "euclidean":
                W, H, obj_val = self.update_euclidean(W, H, V)
            else:
                W, H, obj_val = self.update_divergence(W, H, V)
            obj_vals.append(obj_val)
            print("Iteration %d; objective value = %.2f" % (iter, obj_val))
        if return_obj_values:
            return(W, H, obj_vals)
        return(W, H)
