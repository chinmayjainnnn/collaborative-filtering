import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

def CUR_with_batching(matrix, k, batch_size=500):
    '''
        matrix : (m x n )
        k : dimension to reduce
        batch_size : number of rows/columns to process at once
        Output: C(m x k), U(k x k), R(k x n)
    '''
    m, n = matrix.shape
    
    if k > m or k > n:
        print("No possible CUR decomposition as k is greater than m or n")
        return None, None, None
    
    # Selecting columns in batches
    def columnSelectWithBatch(A, k, batch_size):
        m, n = A.shape
        A_sq_el = A**2
        d = np.sum(A_sq_el)
        l2_norm = np.sum(A_sq_el, axis=0)  # l2 norm square for each column
        prob_col = l2_norm / d
        
        columns = []
        col_indices = []
        B = A.T
        
        for i in range(0, n, batch_size):
            batch_probs = prob_col[i:i+batch_size]
            top_indices = np.argsort(-batch_probs)[:k] + i
            col_indices.extend(top_indices)
            columns.append(B[top_indices])
        
        return np.hstack(columns).T, col_indices[:k]

    W = np.zeros((k, k))
    C, cols = columnSelectWithBatch(matrix, k, batch_size)
    R, rows = columnSelectWithBatch(matrix.T, k, batch_size)
    
    for i in range(len(rows)):
        for j in range(len(cols)):
            W[i][j] = matrix[rows[i], cols[j]]
    
    U = np.linalg.pinv(W)
    R = R.T
    
    return C, U, R


def matrix_factorization_with_batching(matrix, P, Q, K, mx_itr=1000, alpha=0.003, lmda=0.05, batch_size=500):
    '''
    Matrix factorization with batching
    matrix : rating matrix
    P: n_users * K
    Q: n_movies * K
    K: latent features
    batch_size: number of users/movies to process at a time
    '''
    n_users, n_items = matrix.shape
    epoch = 0
    
    while epoch < mx_itr:
        # Shuffle the users
        user_indices = np.arange(n_users)
        np.random.shuffle(user_indices)
        
        for batch_start in range(0, n_users, batch_size):
            batch_end = min(batch_start + batch_size, n_users)
            batch_users = user_indices[batch_start:batch_end]
            
            for i in batch_users:
                for j in range(n_items):
                    if matrix[i][j] > 0:
                        # Error for ith user and jth movie latent vector
                        error_ij = matrix[i][j] - np.dot(P[i, :], Q[j, :])

                        for k in range(K):
                            # Gradient update
                            P[i][k] += alpha * (2 * error_ij * Q[j][k] - lmda * P[i][k])
                            Q[j][k] += alpha * (2 * error_ij * P[i][k] - lmda * Q[j][k])

        # Calculate loss
        mt = np.dot(P, Q.T)
        loss = np.linalg.norm(matrix - mt)
        epoch += 1
        print(f"Total loss is {loss} after {epoch} iterations", end='\r')

        if loss < 1:
            return P, Q
    
    return P, Q


def get_matrix(X, y):
    users = np.array(X.userId.values)
    movies = np.array(X.movieId.values)
    ratings = np.array(y)
    movieToIndex = sorted(set(movies))
    d = {}
    for i in range(len(movieToIndex)):
        d[movieToIndex[i]] = i
    matrix = np.zeros((users.max() + 1, len(movieToIndex)))
    for i in range(X.shape[0]):
        matrix[users[i]][d[movies[i]]] = ratings[i]
    return matrix, d


def run_dataset(dataset_path, dataset="small"):
    df = pd.read_csv(dataset_path)
    X = df[['userId', 'movieId']]
    y = df['rating']
    matrix, _ = get_matrix(X, y)
    
    print("\n--------------------------------  CUR decomposition with Batching ------------------------------\n")
    C, U, R = CUR_with_batching(matrix, k=50, batch_size=100)
    if C is not None and U is not None and R is not None:
        reconstructed_matrix = np.matmul(C, np.matmul(U, R))
        cur_loss = np.linalg.norm(matrix - reconstructed_matrix)
        print(f"CUR decomposition loss: {cur_loss}")
    
    print("\n--------------------------------  Matrix Factorization with Batching ------------------------------\n")
    k = 30
    P = np.random.rand(matrix.shape[0], k)
    Q = np.random.rand(matrix.shape[1], k)
    P, Q = matrix_factorization_with_batching(matrix, P, Q, K=k, mx_itr=1000, batch_size=100)
    if P is not None and Q is not None:
        reconstructed_matrix = np.matmul(P, Q.T)
        mf_loss = np.linalg.norm(matrix - reconstructed_matrix)
        print(f"Matrix Factorization loss: {mf_loss}")


if __name__ == '__main__':
    dataset_path = 'data/ratings.csv'  # Update this with the path to your dataset
    run_dataset(dataset_path)
