import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


def Plot_ROC():
    an = 1
    if an == 1:
        learnper = [0.45, 0.55, 0.65, 0.75, 0.85]
        Varie =[[0.14, 0.16, 0.10, 0.12, 0.08],
             [0.09, 0.11, 0.08, 0.07, 0.06]]
        roc_score=[]
        roc_act=[]
        for n in range(2): # For all datasets
            Score=[]
            Tar=[]
            y = np.random.randint(low=0, high=2, size=[90, 1])
            y = label_binarize(y, classes=[0, 1])
            ta=y.copy()
            index_1 = np.where(y == 1)
            index_0 = np.where(y == 0)
            y = y.astype('float')
            y[index_1] = (np.random.randint(low=600, high=980, size=len(index_1[0]))/1000)
            y[index_0] = (np.random.randint(low=200, high=700, size=len(index_0[0])) / 1000)
            for i in range(5):
                y_score = y.copy().astype('float')
                varie = Varie[1][i] + ((Varie[0][i] - Varie[1][i]) / len(learnper)) * (len(learnper) - 4)
                perc_1 = round(index_1[0].shape[0] * varie)
                perc_0 = round(index_0[0].shape[0] * varie)
                rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                y_score[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = (np.random.randint(low=100, high=500, size=perc_1)/1000)#+np.random.rand(1, perc_1)
                y_score[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = np.random.randint(low=550, high=980, size=perc_0)/1000 # + +np.random.rand(1, perc_0)
                Score.append(y_score)
                Tar.append(ta)
            roc_score.append(Score)
            roc_act.append(Tar)
        np.save('roc_score.npy', roc_score)
        np.save('roc_act.npy', roc_act)

Plot_ROC()