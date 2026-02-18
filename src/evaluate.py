from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score

def confuse(target, predictions):
    """
    calculates a confusion matrix using target, predictions. returns confusion_matrix(2x2 array), disp(plottable matrix), stats(dict)
    """
    confusion_matrix = metrics.confusion_matrix(target[1],predictions)

    stats = [accuracy_score(target[1],predictions),
             precision_score(target[1],predictions),
             recall_score(target[1],predictions),
             ]

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Class 0', 'Class 1']) 
    return confusion_matrix, disp, stats
