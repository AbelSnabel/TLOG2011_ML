import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def split(features, target):
    """
    splits df ~80/20 training/test, returns [xtr,xte],[ytr,yte]
    """
    xtr, xte, ytr, yte = train_test_split(features, target, random_state=42,train_size=0.8)
    return [xtr,xte],[ytr,yte]

def initializeLG(features, target):
    """
    initializes logreg model using features, target. returns the initialized and fitted logreg model
    """
    lg = LogisticRegression(verbose=False,max_iter=1500)

    lg.fit(features[0], target[0])

    return lg

def predicateLG(features, target, lg):
    """
    predicts values for test data using features, target, logreg. returns predictions. 
    """
    predictions = lg.predict(features[1])
    return predictions

def initializeRTC(features, target):
    """
    initializes dtc model using features, target. returns the initialized and fitted dtc model
    """
    rtc = RandomForestClassifier(n_estimators=100, random_state=42)

    rtc.fit(features[0], target[0])

    return rtc

def predicateRTC(features, target, rtc):
    """
    predicts values for test data using features, target, dtc. returns predictions. 
    """
    predictions = rtc.predict(features[1])
    return predictions


def initializeXGB(features, target):
    scale_pos_weight = (target[0] == 0).sum() / (target[0] == 1).sum()

    print(scale_pos_weight)

    xgb = XGBClassifier(n_estimators=2500, max_depth=5, learning_rate=0.08, objective='binary:logistic', scale_pos_weight = scale_pos_weight, eval_metric = "rmse")

    xgb.fit(features[0], target[0])

    return xgb

def predicateXGB(features, target, xgb):
    predictions = xgb.predict(features[1])

    return predictions