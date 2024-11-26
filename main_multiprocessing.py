import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")


def CUR(matrix, k):
    """
    matrix : (m x n )
    k : dimension to reduce
    Output: C(m x k), U(k x k), R(k x n)
    """
    m, n = matrix.shape
    if k > m or k > n:
        print("No possible CUR decomposition as k is greater than m, n")
        return None, None, None

    def pseudoInverse(A):
        U, sigma, VT = np.linalg.svd(A, full_matrices=False)
        igma = np.array([1 / item if item > 0.1 else 0 for item in sigma])
        V = VT.T
        UT = U.T
        return np.matmul(np.matmul(V, np.diag(igma)), UT)

    def columnSelect(A, k):
        A_sq_el = A**2
        d = np.sum(A_sq_el)
        l2_norm = np.sum(A_sq_el, axis=0)
        prob_col = l2_norm / d

        d = {prob_col[i]: i for i in range(len(prob_col))}

        cols = []
        columns = []
        B = A.T
        for item in sorted(d.keys(), reverse=True):
            if k:
                k -= 1
                cols.append(B[d[item]])
                columns.append(d[item])
        return np.array(cols).T, columns

    W = np.zeros((k, k))
    C, cols = columnSelect(matrix, k)
    R, rows = columnSelect(matrix.T, k)
    for i in range(len(rows)):
        for j in range(len(cols)):
            W[i][j] = matrix[rows[i]][cols[j]]
    R = R.T

    U = pseudoInverse(W)

    return C, U, R


def matrix_factorization(matrix, P, Q, K, mx_itr=1000, alpha=0.003, lmda=0.05):
    epoch = 0
    while epoch < mx_itr:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] > 0:
                    error_ij = matrix[i][j] - np.dot(P[i, :], Q[j, :])

                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * error_ij * Q[j][k] - lmda * P[i][k])
                        Q[j][k] = Q[j][k] + alpha * (2 * error_ij * P[i][k] - lmda * Q[j][k])

        mt = np.dot(P, Q.T)
        loss = np.linalg.norm(matrix - mt)

        epoch += 1
        print("Total loss is {} after {} iterations ".format(loss, epoch), end="\r")
        if loss < 1:
            break

    return P, Q


def get_matrix(X, y):
    users = np.array(X.userId.values)
    movies = np.array(X.movieId.values)
    ratings = np.array(y)
    movieToIndex = sorted(set(movies))
    d = {movieToIndex[i]: i for i in range(len(movieToIndex))}
    matrix = np.zeros((users.max() + 1, len(movieToIndex)))
    for i in range(X.shape[0]):
        matrix[users[i]][d[movies[i]]] = ratings[i]
    return matrix, d


def CUR_task(matrix, max_k=500):
    cur_loss = []
    for i in range(1, max_k + 1):
        C, U, R = CUR(matrix, i)
        B = np.matmul(C, np.matmul(U, R))
        cur_loss.append(np.linalg.norm(matrix - B))
    return cur_loss


def PQ_task(matrix_t, X_train, X_test, y_train, y_test, k=30):
    P = np.random.rand(matrix_t.shape[0], k)
    Q = np.random.rand(matrix_t.shape[1], k)

    P, Q = matrix_factorization(matrix_t, P, Q, k, mx_itr=500, alpha=0.03, lmda=0.5)
    mt = np.matmul(P, Q.T)
    total_error = np.linalg.norm(matrix_t - mt)

    print("Total loss for PQ decomposition of matrix: {}".format(total_error))

    train_err = 0
    movie_map = {v: k for k, v in enumerate(np.unique(X_train.movieId))}
    for item in np.array(X_train):
        train_err += (matrix_t[item[0]][movie_map[item[1]]] - mt[item[0]][movie_map[item[1]]]) ** 2

    print("\nTrain MSE Loss: {}".format(train_err / X_train.shape[0]))

    test_err = 0
    for i in range(len(X_test)):
        user, movie = X_test.iloc[i]
        if movie in movie_map:
            test_err += (y_test.iloc[i] - mt[user][movie_map[movie]]) ** 2

    print("\nTest MSE Loss: {}".format(test_err / len(X_test)))


def run_dataset(dataset_path, dataset="small"):
    df = pd.read_csv(dataset_path)
    X = df[["userId", "movieId"]]
    y = df["rating"]

    matrix, _ = get_matrix(X, y)

    if dataset == "small":
        # Parallelize CUR and PQ computations
        with Pool(cpu_count()) as pool:
            cur_result = pool.apply_async(CUR_task, (matrix,))
            pq_result = pool.apply_async(PQ_task, (matrix, X, y, X, y))

            cur_loss = cur_result.get()
            pq_result.get()

            plt.figure(figsize=(15, 10))
            plt.plot(range(1, len(cur_loss) + 1), cur_loss)
            plt.xlabel("k (latent factors)")
            plt.ylabel("Loss of data")
            plt.title("CUR Loss")
            plt.show()


if __name__ == "__main__":
    run_dataset(dataset_path="data/ratings.csv")
