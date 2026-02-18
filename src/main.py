from preprocessing import get_preprocess
from features import get_features
from train import split, initialize, predicate
from evaluate import confuse

import matplotlib.pyplot as plt

df = get_preprocess()
features, target = get_features(df)

features = split(features)
target = split(target)

logreg = initialize(features, target)

predictions = predicate(features, target, logreg)

confusion_matrix, disp_confusion_matrix = confuse(target, predictions)

disp_confusion_matrix.plot()
plt.show()