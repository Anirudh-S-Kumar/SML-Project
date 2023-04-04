from pipeline_components import *

class Pipeline:
    def __init__(self, clustering_alg: str, dim_reduction_alg: str, outlier_detection_alg: str, classification_alg: str):
        self.clustering_alg = clustering_alg
        self.dim_reduction_alg = dim_reduction_alg
        self.outlier_detection_alg = outlier_detection_alg
        self.classification_alg = classification_alg

    def fit(self, X_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # clustering
        if self.clustering_alg:
            cl = Clustering(self.clustering_alg, 2)
            X_t, y_t = cl.transform(X_t, y_t)

        # dimensionality reduction
        if self.dim_reduction_alg:
            dr = DimReduction(self.dim_reduction_alg, 2)
            X_t, y_t = dr.transform(X_t, y_t)

        # outlier detection
        if self.outlier_detection_alg:
            od = OutlierDetection(self.outlier_detection_alg)
            X_t, y_t = od.transform(X_t, y_t)

        # classification
        cl = Classification(self.classification_alg)

        return X_t, y_t