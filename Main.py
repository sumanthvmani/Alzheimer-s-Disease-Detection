import numpy as np
import os
from numpy import matlib
from sklearn.utils import shuffle
import cv2 as cv
from BES import BES
from BMO import BMO
from FFO import FFO
from Global_Vars import Global_Vars
from MOA import MOA
from Model_ASE_A_ConvTNet import Model_ASE_A_ConvTNet
from Model_CNN import Model_CNN
from Model_Randomforest import Model_Randomforest
from Model_XGBoost import Model_XGBoost
from PROPOSED import PROPOSED
from PSO import PSO
from Plot_Results import *
from objfun import objfun_cls


no_of_dataset = 2

# Read dataset
an = 0
if an == 1:
    for i in range(2):
        Images = []
        label = []
        dir = f'./Dataset/Dataset{i + 1}/'
        dir_list = os.listdir(dir)
        for j in range(len(dir_list)):
            file = dir + dir_list[j] + '/'
            file_list = os.listdir(file)
            for k in range(len(file_list)):
                print(i, j, k)
                file1 = file + file_list[k]
                img = cv.imread(file1)
                if len(img.shape) == 3:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.resize(img, (64, 64))
                file_name = dir_list[j]
                label.append(file_name)
                Images.append(img)
        label = np.asarray(label)
        uni = np.unique(label)
        tar = np.zeros((len(label), len(uni))).astype('int')
        for l in range(len(uni)):
            ind = np.where(uni[l] == label)
            tar[ind[0], l] = 1
        tar = shuffle(tar)
        np.save(f'Images_{i + 1}.npy', Images)
        np.save(f'Target_{i+1}.npy',tar)

# Optimization for Classification
an = 0
if an == 1:
    Fitness = []
    Best_solution = []
    for n in range(no_of_dataset):
        Feat = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the images
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Learning Rate, step per Epoch in ConvTNet
        xmin = matlib.repmat(np.asarray([5, 0.01, 100]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 0.99, 500]), Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("BES...")
        [bestfit1, fitness1, bestsol1, time1] = BES(initsol, fname, xmin, xmax, Max_iter)  # BES

        print("BMO...")
        [bestfit2, fitness2, bestsol2, time2] = BMO(initsol, fname, xmin, xmax, Max_iter)  # BMO

        print("MOA...")
        [bestfit3, fitness3, bestsol3, time3] = MOA(initsol, fname, xmin, xmax, Max_iter)  # MOA

        print("FFO...")
        [bestfit4, fitness4, bestsol4, time4] = FFO(initsol, fname, xmin, xmax, Max_iter)  # FFO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED
        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
        Fitness.append(fitness)
        Best_solution.append(BestSol_CLS)
    np.save('Fitness.npy', np.asarray(Fitness))
    np.save('BestSol_CLS.npy', np.asarray(Best_solution))  # Best solution


# Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feat = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the images
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]
        k_fold = 5
        Per = 1 / k_fold
        EVAL = []
        Perc = round(Feat.shape[0] * Per)
        for i in range(k_fold):
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval = np.zeros((10, 23))
            for j in range(BestSol.shape[0]):
                print(j)
                sol = BestSol[j, :]
                Eval[j, :], pred0 = Model_ASE_A_ConvTNet(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
            Eval[5, :], pred1 = Model_Randomforest(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred2 = Model_XGBoost(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred3 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred4 = Model_ASE_A_ConvTNet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_KFold.npy', np.asarray(Eval_all))

Plot_Fitness()
Plot_ROC()
Plot_Results()
Plot_table()
new_plot()

