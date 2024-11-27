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

## Results

**Top 20 singular values are**
[9032.38102201 4265.13020478 2962.83432586 2856.37494764 2441.34461236
 2269.55931732 2169.8992637  1848.47223494 1701.69413469 1528.15832014
 1476.74413397 1449.77168211 1432.00119537 1413.20720491 1319.28764566
 1281.82058619 1213.72797731 1203.99401803 1198.55552737 1135.24246017]

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


