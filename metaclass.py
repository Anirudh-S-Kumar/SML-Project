import pipeline_components as pc
import pandas as pd
import numpy as np

class Pipeline:
    def __init__(self, 
            clustering_alg: str, 
            dim_reduction_alg_1: str, 
            dim_reduction_alg_2: str,
            outlier_detection_alg: str, 
            classification_alg: str, 
            ensemble_alg: str = None
            ):
        
        self.clustering_alg = clustering_alg
        self.dim_reduction_alg_1 = dim_reduction_alg_1
        self.dim_reduction_alg_2 = dim_reduction_alg_2
        self.outlier_detection_alg = outlier_detection_alg
        self.classification_alg = classification_alg
        self.ensemble_alg = ensemble_alg

        self.standardizer = None
        self.cl = None
        self.dim_red_1 = None
        self.dim_red_2 = None

    # def standardize(X_t: np.ndarray) -> np.ndarray:
    #     from sklearn.preprocessing import StandardScaler
    #     scaler = StandardScaler()
    #     X_t = scaler.fit(X_t)
    #     self.standardizer = scaler
    #     return X_t

    def fit(self, X_t: np.ndarray, y_t: np.ndarray) -> None:
        # standardize data
        print("Standardizing data...")
        self.standardizer = pc.Standardizer()
        self.standardizer.fit(X_t)
        X_t = self.standardizer.transform(X_t)

        # clustering
        print(f"Currently at clustering: {self.clustering_alg}")
        if self.clustering_alg:
            cl = pc.Clustering(self.clustering_alg, 20)
            X_t, y_t = cl.transform(X_t, y_t)

        # dimensionality reduction
        print(f"Currently at dim reduction: {self.dim_reduction_alg_1}")
        if self.dim_reduction_alg_1:
            dr = pc.DimReduction(self.dim_reduction_alg_1, 100)
            X_t = dr.transform(X_t, y_t)
            self.dim_red_1 = dr
        print(f"Currently at dim reduction: {self.dim_reduction_alg_2}")
        if self.dim_reduction_alg_2:
            dr = pc.DimReduction(self.dim_reduction_alg_2, 16)
            X_t = dr.transform(X_t, y_t)
            self.dim_red_2 = dr

        # outlier detection
        print(f"Currently at outlier removal: {self.outlier_detection_alg}")
        if self.outlier_detection_alg:
            od = pc.OutlierDetection(self.outlier_detection_alg)
            X_t, y_t = od.transform(X_t, y_t)

        # classification
        print(f"Currently at classifier: {self.classification_alg}")
        clfs = pc.Classification(self.classification_alg)
        clfs.fit(X_t, y_t)
        self.cl = clfs.return_model()

        # ensemble
        print(f"Currently at endsembling: {self.ensemble_alg}")
        if self.ensemble_alg:
            ensemble = pc.Ensemble(self.cl, self.ensemble_alg)
            ensemble.fit(X_t, y_t)
            self.cl = ensemble.return_model()

        print("Done!")
    
    def predict(self, X_t: np.ndarray) -> np.ndarray:
        return self.cl.predict(X_t)
    
    def score(self, X_t: np.ndarray, y_t: np.ndarray) -> float:
        return self.cl.score(X_t, y_t)
    
    def cross_validate(self, X_t: np.ndarray, y_t: np.ndarray, n_splits: int = 5) -> np.ndarray:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(self.cl, X_t, y_t, cv=n_splits)
        return (scores.mean(), scores.std())
    
    def generate_submission(self, X_s: pd.DataFrame) -> None:
        from datetime import datetime

        X_test = X_s.drop(['ID'], axis=1)

        X_test = self.standardizer.transform(X_test)
        # X_test = standardize(X_test)
        
        if self.dim_reduction_alg_1:
            X_test = self.dim_red_1.dim_red.transform(X_test)
        if self.dim_reduction_alg_2:
            X_test = self.dim_red_2.dim_red.transform(X_test)
            # X_test = pc.DimReduction(self.dim_reduction_alg, 100).transform(X_test, None)

        y_pred = self.predict(X_test)

        submission = pd.DataFrame({'ID': X_s['ID'], 'Category': y_pred})
        submission.to_csv(f"submissions/submission_{(datetime.now()).strftime('%Y_%m_%d-%H_%M')}.csv", index=False)


if __name__ == "__main__":
    pipeline = Pipeline(
        clustering_alg=None,
        dim_reduction_alg="lda",
        outlier_detection_alg="isolation_forest",
        classification_alg="rf",
        ensemble_alg="bagging"
    )

    train_data = pd.read_csv('data/train.csv')

    X_t = train_data.drop(['category', 'ID'], axis=1)
    y_t = train_data['category']

    pipeline.fit(X_t, y_t)

    print(pipeline.cross_validate(X_t, y_t))