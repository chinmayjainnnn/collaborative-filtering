# Collaborative Filtering using SVD, CUR, and PQ Matrix Decomposition

This project demonstrates collaborative filtering for recommendation systems using three matrix decomposition techniques: Singular Value Decomposition (SVD), CUR decomposition, and PQ decomposition (Matrix Factorization). These techniques help in dimensionality reduction and latent feature extraction, improving the scalability and accuracy of recommendation systems.

---

## Methods

- **SVD Decomposition**: Computes singular values and explores the loss of data with varying latent factors.
- **CUR Decomposition**: Approximates the original matrix using selected columns and rows, with tunable latent dimensions.
- **PQ Matrix Factorization**: Learns user and item latent vectors using gradient descent to minimize prediction error.

---

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

To install them, you can run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Dataset

The project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) for movie ratings:

- **Path**: Place the dataset in the `data/ratings.csv`.
- **Structure**: The dataset should include `userId`, `movieId`, and `rating` columns.

---

## How to Run

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/collaborative-filtering.git
    cd collaborative-filtering
    ```

2. Place the dataset file (`ratings.csv`) in the `data/ml-latest-small/` directory.

3. Run the script:

    ```bash
    python collaborative_filtering.py
    ```

---

## Outputs

### 1. **SVD Decomposition**
- Displays the top 20 singular values.
- Plots the loss of data against the number of latent factors (`k`).
- Time taken for SVD decomposition is logged.

### 2. **CUR Decomposition**
- Computes CUR approximation of the matrix.
- Plots the reconstruction loss for varying `k` (latent factors).
- Time taken for CUR decomposition is logged.

### 3. **PQ Decomposition**
- Performs matrix factorization using gradient descent.
- Logs training and test mean squared errors (MSE).
- Time taken for PQ decomposition is logged.

---

<!-- ## Code Highlights

- **SVD Decomposition**: Uses `numpy.linalg.svd` for efficient singular value computation.
- **CUR Decomposition**:
  - Selects important columns and rows based on probabilities derived from norms.
  - Constructs `U` using pseudo-inverse of the core matrix.
- **PQ Decomposition**:
  - Implements gradient updates for user (`P`) and item (`Q`) latent vectors.
  - Regularization ensures better generalization to unseen data.

--- -->

## Visualization

Two key plots are generated:

1. **Loss vs. Latent Factors (SVD)**:
    - Visualizes data reconstruction loss as latent factors (`k`) increase.

2. **Loss vs. Latent Factors (CUR)**:
    - Visualizes CUR reconstruction loss for varying `k`.

---


