import numpy as np
import pandas as pd
import metaclass as mc
import pipeline_components as pc

train_data = pd.read_csv('data/train.csv')

X_t = train_data.drop(['category', 'ID'], axis=1)
y_t = train_data['category']

test_data = pd.read_csv('data/test.csv')

pipeline = mc.Pipeline(
    clustering_alg=("kmeans", 20),
    dim_reduction_algs=[("pca", 119)],
    outlier_detection_alg=None,
    classification_alg="logistic",
    ensemble_algs=None,
)
pipeline.fit(X_t, y_t)
print("Pipeline done")
cv_scores = pipeline.cross_validate(X_t, y_t)
print(f"Cross validation scores: {cv_scores}")

pipeline.generate_submission(test_data)