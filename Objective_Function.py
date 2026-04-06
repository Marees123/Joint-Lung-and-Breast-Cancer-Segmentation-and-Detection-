import numpy as np
from Evaluation import net_evaluation, evaluation
from Global_Vars import Global_Vars
from Model_DA_ViT_UNetPP import Model_DA_ViT_UNetPP
from Model_ERMSC_ConvNeXtV2 import Model_ERMSC_ConvNeXtV2


def objfun(Soln):
    Images = Global_Vars.Images
    GT = Global_Vars.GT
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, Image = Model_DA_ViT_UNetPP(Images, GT, sol)
            Eval = net_evaluation(Images[0], Image[0])
            Fitn[i] = 1 / Eval[5]
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, Image = Model_DA_ViT_UNetPP(Images, GT, sol)
        Eval = net_evaluation(Images[0], Image[0])
        Fitn = 1 / Eval[5]
        return Fitn


def objective_function(Soln):
    Breast_cancer = Global_Vars.Breast_cancer
    Lung_Cancer = Global_Vars.Lung_Cancer
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Lung_Cancer.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Lung_Cancer[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Lung_Cancer[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_ERMSC_ConvNeXtV2(Breast_cancer, Lung_Cancer, Tar, sol)
            Eval = evaluation(Test_Target, pred)
            Fitn[i] = (1 / Eval[7]) + Eval[14]
        return Fitn
    else:
        learnper = round(Lung_Cancer.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Lung_Cancer[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Lung_Cancer[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_ERMSC_ConvNeXtV2(Breast_cancer, Lung_Cancer, Tar, sol)
        Eval = evaluation(Test_Target, pred)
        Fitn = (1 / Eval[7]) + Eval[14]
        return Fitn
