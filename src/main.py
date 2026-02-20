from preprocessing import get_preprocess, random_undersample
from features import get_features
from train import split, initializeLG, predicateLG, initializeRTC, predicateRTC, initializeXGB, predicateXGB
from evaluate import confuse, auc
from sklearn import tree
import matplotlib.pyplot as plt

df = get_preprocess()
features, target = get_features(df)

#features, target = random_undersample(features, target)

features.to_csv(r'data\features.csv', index=False)

features, target = split(features, target)

""""
logreg = initializeLG(features, target)

predictionsLG = predicateLG(features, target, logreg)

confusion_matrix, disp_confusion_matrix, statistics = confuse(target, predictionsLG)

disp_confusion_matrix.plot()
plt.show()
print("acc, prec, rec equals", statistics)
"""

XGB = initializeXGB(features, target)

predictionsXGB = predicateXGB(features, target, XGB)

confusion_matrix, disp_confusion_matrix, statistics = confuse(target, predictionsXGB)

disp_confusion_matrix.plot()
plt.show()
print("acc, prec, rec equals", statistics)