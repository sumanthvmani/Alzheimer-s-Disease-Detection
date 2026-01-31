import numpy as np
from Evaluation_nrml import evaluation
from Global_Vars import Global_Vars
from Model_ASE_A_ConvTNet import Model_ASE_A_ConvTNet


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_ASE_A_ConvTNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(Test_Target, pred)
            Fitn[i] = (1 / Eval[15]) + Eval[11]  # (1 / CSI) + FDR
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_ASE_A_ConvTNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(Test_Target, pred)
        Fitn = (1 / Eval[15]) + Eval[11]  # (1 / CSI) + FDR

        return Fitn

