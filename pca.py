import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        """
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
		    Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
		    X: (N,D) numpy array corresponding to a dataset
		
		Return:
		    None
		
		Set:
		    self.U: (N, min(N,D)) numpy array
		    self.S: (min(N,D), ) numpy array
		    self.V: (min(N,D), D) numpy array
		"""
        X = np.array(X)
        X_centered = X - np.mean(X, axis=0, keepdims = True)
        U,S,V = np.linalg.svd(X_centered, full_matrices = False)
        self.U = U
        self.S = S
        self.V = V
        return

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		by projecting onto the principal components.
		Utilize class members that were previously set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    K: int value for number of columns to be kept
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        data = np.array(data)
        centered_data = data - np.mean(data, axis=0, keepdims = True)
        V = self.V[:K, :]
        X_new = np.dot(centered_data, V.T)
        return X_new

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    retained_variance: float value for amount of variance to be retained
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
		           to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        data = np.array(data)
        og_sum = np.sum(self.S**2)
        S = np.cumsum(self.S**2)
        k = 0
        curr_ratio = S[k]/og_sum
        while curr_ratio < retained_variance:
            k = k+1
            curr_ratio = S[k]/og_sum
        k = k+1
        centered_data = data - np.mean(data, axis=0, keepdims = True)
        V = self.V[:k, :]
        X_new = np.dot(centered_data, V.T)
        return X_new

    def get_V(self) ->np.ndarray:
        """
		Getter function for value of V
		"""
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
        """
		You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
		For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
		You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using matplotlib.
		
		Args
		    xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
		    ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""
        pca = PCA()
        pca.fit(X)
        x_pca = pca.transform(X, K=3)

		# ===== 2D scatterplot using top 2 components =====
        plt.figure(figsize=(6, 5))
        for label in np.unique(y):
            plt.scatter(
				x_pca[y == label, 0],
				x_pca[y == label, 1],
				label=f"Label {label}",
				alpha=0.7
			)
        plt.title("2D PCA - Top 2 Components")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

		# ===== 3D scatterplot using top 3 components =====
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        for label in np.unique(y):
            ax.scatter(
				x_pca[y == label, 0],
				x_pca[y == label, 1],
				x_pca[y == label, 2],
				label=f"Label {label}",
				alpha=0.7
			)
        ax.set_title("3D PCA - Top 3 Components")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.tight_layout()
        plt.show()

		# ===== 2D scatterplot using two random components =====
        D = x_pca.shape[1]
        random_indices = np.random.choice(D, size=2, replace=False)
        plt.figure(figsize=(6, 5))
        for label in np.unique(y):
            plt.scatter(
				x_pca[y == label, random_indices[0]],
				x_pca[y == label, random_indices[1]],
				label=f"Label {label}",
				alpha=0.7
			)
        plt.title(f"2D PCA - Random Components ({random_indices[0]+1}, {random_indices[1]+1})")
        plt.xlabel(f"PC{random_indices[0]+1}")
        plt.ylabel(f"PC{random_indices[1]+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
