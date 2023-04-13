from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from pipeline_components import *
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import metaclass as mc
import pipeline_components as pc

train_data = pd.read_csv('data/train.csv')

X_t = train_data.drop(['category', 'ID'], axis=1)
y_t = train_data['category']



if __name__ == "__main__":
    pipeline = Pipeline([
        ("PCA 300", PCA(n_components=300)),
        ("LDA 19", LinearDiscriminantAnalysis(n_components=19)),
        ('Bagging', BaggingClassifier(base_estimator=MLPClassifier(activation='relu', 
                                                                    solver='lbfgs', 
                                                                    alpha=10, 
                                                                    hidden_layer_sizes=(310), 
                                                                    random_state=1,
                                                                    max_iter=1000), 
                                    n_estimators=10, random_state=1))
      
        ])

    pipeline.fit(X_t, y_t)
    print("Pipeline done")
    cross_val_scores=cross_val_score(pipeline, X_t, y_t, cv=5)
    print(cross_val_scores.mean(), cross_val_scores.std())
