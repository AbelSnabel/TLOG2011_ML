import numpy as np
from sklearn.linear_model import LogisticRegression


def split(df):
    """
    splits df ~80/20 training/test
    should be changed to random 80%/20% later 
    """
    x1 = df[0:int(len(df)*0.80)]
    x2 = df[int(len(df)*0.80):]
    return [x1,x2]

def initialize(features, target):
    """
    initializes logreg model using features, target. returns the initialized and fitted logreg model
    """
    lg = LogisticRegression(verbose=True)

    lg.fit(features[0], target[0])

    return lg

def predicate(features, target, lg):
    """
    predicts values for test data using features, target, logreg. returns predictions. 
    """
    predictions = lg.predict(features[1])
    return predictions
