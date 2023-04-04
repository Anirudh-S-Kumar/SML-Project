import numpy as np

class OutlierDetection:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.mapping = {
            "isolation_forest": self.isolation_forest,
            "lof": self.lof
        }

    def transform(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.mapping[self.algorithm](X_t, y_t)

    def isolation_forest(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest().fit(X_t)
        y_pred = clf.predict(X_t)

        # percentage of outliers
        print("Outlier percentage:", sum(y_pred == -1) / len(y_pred))

        # removing outliers
        X_t_iso = X_t[y_pred == 1]
        y_t_iso = y_t[y_pred == 1]

        return X_t_iso, y_t_iso

    def lof(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import LocalOutlierFactor

        # create a LOF object
        clf = LocalOutlierFactor(n_neighbors=20)

        # fit the model
        y_pred = clf.fit_predict(X_t)

        # percentage of outliers
        print("Outlier percentage:", sum(y_pred == -1) / len(y_pred))

        # removing outliers
        X_t_lof = X_t[y_pred == 1]
        y_t_lof = y_t[y_pred == 1]

        return X_t_lof, y_t_lof


class DimReduction:
    def __init__(self, algorithm: str, n_components: int):
        self.algorithm = algorithm
        self.n_components = n_components
        self.mapping = {
            "pca": self.pca,
            "lda": self.lda
        }

    def transform(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.mapping[self.algorithm](X_t, y_t)

    def pca(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_components)
        X_t_pca = pca.fit_transform(X_t)

        return X_t_pca, y_t
    
    def lda(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis()
        X_t_lda = lda.fit_transform(X_t, y_t)

        return X_t_lda, y_t