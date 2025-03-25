import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

def plot_pca(X):
    """
    Perform PCA using SVD and plot the variance explained.

    PARAMETERS
    ----------
    X : ndarray
        Feature matrix (samples × features).
    threshold : float
        The cumulative variance threshold to mark on the plot.

    RETURNS
    -------
    None
    """
    # 1. Standardize data (zero mean and unit variance)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 2. Center the data (mean subtraction)
    Y = X_std - np.mean(X_std, axis=0)

    # 3. Compute SVD (PCA decomposition)
    U, S, Vt = svd(Y, full_matrices=False)

    # 4. Compute variance explained by each principal component
    rho = (S**2) / np.sum(S**2)  # Variance explained

    # 5. Plot variance explained
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative")
    
    # Labels and title
    plt.title("Variance Explained by Principal Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.legend()
    plt.grid()

    # Show plot
    plt.show()