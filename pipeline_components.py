import numpy as np
from typing import Literal

class Standardizer:
    def __init__(self):
        self.scaler = None

    def fit(self, X_t: np.ndarray) -> None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_t = scaler.fit(X_t)
        self.scaler = scaler

    def transform(self, X_t: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X_t)
    
class Clustering:
    def __init__(self, algorithm: Literal['kmeans', 'agglomerative'], n_clusters: int):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.mapping = {
            "kmeans": self.kmeans,
            "agglomerative": self.agglomerative
        }
        self.cl = None

    def transform(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.mapping[self.algorithm](X_t, y_t)

    def kmeans(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X_t)
        self.cl = kmeans
        y_pred = kmeans.predict(X_t)
        # Use the cluster labels as new feature values
        print(f"Shape of X_t: {X_t.shape}")
        print(f"Shape of y_pred: {y_pred.shape}")
        X_t_kmeans = np.c_[X_t, y_pred]
        print(f"Shape of X_t_kmeans: {X_t_kmeans.shape}")

        return X_t_kmeans, y_pred

    def agglomerative(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import AgglomerativeClustering

        agg = AgglomerativeClustering(n_clusters=self.n_clusters).fit(X_t)
        y_pred = agg.labels_

        # removing outliers
        X_t_agg = X_t[y_pred == 1]
        y_t_agg = y_t[y_pred == 1]

        return X_t_agg, y_t_agg


class OutlierDetection:
    def __init__(self, algorithm: Literal['isolation_forest', 'lof']):
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
    def __init__(self, algorithm: Literal['pca', 'lda'], n_components: int):
        self.algorithm = algorithm
        self.n_components = n_components
        self.mapping = {
            "pca": self.pca,
            "lda": self.lda
        }
        self.dim_red = None

    def transform(self, X_t: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        return self.mapping[self.algorithm](X_t, y_t)

    def pca(self, X_t: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_components, random_state=42)
        X_t_pca = pca.fit_transform(X_t)  
        self.dim_red = pca

        return X_t_pca  
    
    def lda(self, X_t: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis(n_components=self.n_components)

        if y_t is not None:
            X_t_lda = lda.fit_transform(X_t, y_t)
        else:
            X_t_lda = lda.fit_transform(X_t)
        self.dim_red = lda

        return X_t_lda
    

class Classification:
    def __init__(self, algorithm: Literal["rf", "knn", "logistic", "mlp", "naive_bayes"]):
        self.algorithm = algorithm
        self.mapping = {
            "rf": self.rf,
            "knn": self.knn,
            "logistic" : self.logistic,
            "mlp": self.mlp,
            "naive_bayes": self.naive_bayes
        }
        self.clf = None

    def rf(self, X_t: np.ndarray, y_t: np.ndarray):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier()
        clf.fit(X_t, y_t)
        return clf

    
    def knn(self, X_t: np.ndarray, y_t: np.ndarray):
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier()
        clf.fit(X_t, y_t)
        return clf

    def logistic(self, X_t: np.ndarray, y_t: np.ndarray):
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=10000, random_state=42)
        clf.fit(X_t, y_t)
        return clf
    
    def naive_bayes(self, X_t: np.ndarray, y_t: np.ndarray):
        from sklearn.naive_bayes import GaussianNB

        clf = GaussianNB()
        clf.fit(X_t, y_t)
        return clf

    
    def mlp(self, X_t: np.ndarray, y_t: np.ndarray):
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(activation='relu', solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(20), random_state=1)
        clf.fit(X_t, y_t)
        print("Number of features:", clf.n_features_in_)
        print("Number of layers:", clf.n_layers_)
        print("Number of outputs:", clf.n_outputs_)
        print("Number of iterations:", clf.n_iter_)
        print("Activation function:", clf.out_activation_)
        print("Class labels:", clf.classes_)
        print("----")
        return clf
        
    def fit(self, X_t: np.ndarray, y_t: np.ndarray):
        self.clf = self.mapping[self.algorithm](X_t, y_t)
    
    def predict(self, X_t: np.ndarray) -> np.ndarray:
        return self.clf.predict(X_t)
    
    def return_model(self):
        return self.clf

class Ensemble:
    def __init__(self, cl, algorithm: Literal["bagging", "adaboost"]):
        self.algorithm = algorithm
        self.cl = cl
        self.mapping = {
            "bagging": self.bagging,
            "adaboost": self.adaboost,
            "xgboost": self.xgboost
        }
        self.ensemble_cl = None


    def bagging(self, X_t: np.ndarray, y_t: np.ndarray):
        from sklearn.ensemble import BaggingClassifier

        clf = BaggingClassifier(self.cl, n_estimators=10)
        clf.fit(X_t, y_t)
        return clf
    
    def adaboost(self, X_t: np.ndarray, y_t: np.ndarray):
        from sklearn.ensemble import AdaBoostClassifier

        clf = AdaBoostClassifier(self.cl, n_estimators=10)
        clf.fit(X_t, y_t)
        return clf
    
    def xgboost(self, X_t: np.ndarray, y_t: np.ndarray):
        from xgboost import XGBClassifier

        clf = XGBClassifier()
        clf.fit(X_t, y_t)
        return clf

    def fit(self, X_t: np.ndarray, y_t: np.ndarray):
        self.ensemble_cl = self.mapping[self.algorithm](X_t, y_t)
    
    def return_model(self):
        return self.ensemble_cl