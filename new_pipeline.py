from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Pipeline:

    def __init__(self,             clustering_alg: tuple[str, int], 
            dim_reduction_algs: list[tuple[str, int]], 
            outlier_detection_alg: str, 
            classification_alg: str, 
            ensemble_algs: list[str] = None):
        self.clustering_alg = clustering_alg
        self.dim_reduction_algs = dim_reduction_algs
        self.outlier_detection_alg = outlier_detection_alg
        self.classification_alg = classification_alg
        self.ensemble_algs = ensemble_algs

        self.standardizer = None
        self.cl = None
        self.clus = None
        self.dim_red_objs = []

    