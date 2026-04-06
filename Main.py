import pydicom as dicom
import numpy as np
import os
import cv2 as cv
import pandas as pd
from numpy import matlib
from CPOA import CPOA
from ECO import ECO
from FOA import FOA
from Global_Vars import Global_Vars
from Model_DA_ViT_UNetPP import Model_DA_ViT_UNetPP
from Model_DCNN import Model_DCNN
from Model_DDA_AttResUNet import Model_DD_Attention_ResUNet
from Model_ERMSC_ConvNeXtV2 import Model_ERMSC_ConvNeXtV2
from Model_FENN import Model_FENN
from Model_RAN import Model_RAN
from Model_ResUnet import Model_ResUNet
from Model_Unet import Model_Unet
from Objective_Function import objfun, objective_function
from Plot_Results import *
from Proposed import Proposed
from SAA import SAA

# Read Breast Cancer Images
an = 0
if an == 1:
    org_path = './Datas/Breast Cancer Dataset/Agu_Images'
    Gt_path = './Datas/Breast Cancer Dataset/Agu_GT'
    out_dir = os.listdir(org_path)
    GT_dir = os.listdir(Gt_path)
    Images = []
    GT = []
    for i in range(len(out_dir)):
        print(i)
        file_name = org_path + '/' + out_dir[i]
        gt_image = Gt_path + '/' + GT_dir[i]
        Image = cv.imread(file_name)
        Image = cv.resize(Image, (512, 512))
        Image = np.uint8(Image)
        gt_img = cv.imread(gt_image)
        gt_img = cv.resize(gt_img, (512, 512))
        gt_img = np.uint8(gt_img)
        Images.append(Image)
        GT.append(gt_img)
    np.save('Images_1.npy', Images)
    np.save('GT_1.npy', GT)


# Read Lung Cancer Images
an = 0
if an == 1:
    Image = []
    Target = []
    path = './Lung Cancer Dataset/Lung-PET-CT-Dx'
    out_dir = os.listdir(path)
    del out_dir[0]
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            subfolder = folder + '/' + in_dir[j]
            dir = os.listdir(subfolder)
            for k in range(len(dir)):
                folders = subfolder + '/' + dir[k]
                sub_dir = os.listdir(folders)
                for m in range(len(sub_dir)):
                    print(i, j, k, m)
                    FileName = folders + '/' + sub_dir[m]
                    ds = dicom.dcmread(FileName)
                    image = (ds.pixel_array / 13).astype('uint8')
                    split_Data = out_dir[i].split('-')
                    if split_Data[1][0] == 'A':
                        Target.append(0)
                    elif split_Data[1][0] == 'B':
                        Target.append(1)
                    elif split_Data[1][0] == 'E':
                        Target.append(2)
                    else:
                        Target.append(3)
                    images = cv.resize(image, [256, 256])
                    Image.append(images)
    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Images_2.npy', Shuffled_Datas)
    np.save('Target.npy', Shuffled_Target)


# Generate Groundtruth for Lung Cancer Images Only
an = 0
if an == 1:
    Images = np.load('Images_2.npy', allow_pickle=True)
    Mask_Img = []
    for j in range(len(Images)):
        print(j, len(Images))
        img = Images[j]
        # Apply Gaussian Blur
        blurred = cv.GaussianBlur(img, (5, 5), 0)
        # Otsu thresholding (binary inverse)
        thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # Connected Components Analysis
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(thresh, connectivity=4, ltype=cv.CV_32S)
        # Initialize blank mask
        output_mask = np.zeros(img.shape, dtype="uint8")
        # Filter components by area and draw them as white
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv.CC_STAT_AREA]
            if 0 < area < 1000:  # Change area limits if needed
                component_mask = (labels == i).astype("uint8") * 255
                output_mask = cv.bitwise_or(output_mask, component_mask)
        Mask_Img.append(output_mask)
    np.save('GT_2.npy', np.asarray(Mask_Img))

no_of_dataset = 2

# Preprocessing
an = 0
if an == 1:
    for m in range(no_of_dataset):
        Images = np.load('Images_'+str(m+1)+'.npy', allow_pickle=True)
        Prep_images = []
        # Create CLAHE object
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for n in range(len(Images)):
            img = Images[n]
            # Step 1: Median Filtering
            median_img = cv.medianBlur(img, 5)   # kernel size = 5
            # Step 2: CLAHE (handle grayscale & color)
            if len(median_img.shape) == 2:
                # Grayscale image
                clahe_img = clahe.apply(median_img)
            else:
                # Color image → convert to LAB
                lab = cv.cvtColor(median_img, cv.COLOR_BGR2LAB)
                l, a, b = cv.split(lab)
                l_clahe = clahe.apply(l)
                lab_clahe = cv.merge((l_clahe, a, b))
                clahe_img = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)
            Prep_images.append(clahe_img)
        np.save('Preprocessed_Images_'+str(m+1)+'.npy', Prep_images)


# Optimization for Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Preprocessed_Images_'+str(n+1)+'.npy', allow_pickle=True)
        GT = np.load('GT_'+str(n+1)+'.npy', allow_pickle=True)
        Global_Vars.Images = Images
        Global_Vars.GT = GT
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Learning Rate and Steps per epoch
        xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = objfun
        max_iter = 50

        print('SAA....')
        [bestfit1, fitness1, bestsol1, Time1] = SAA(initsol, fname, xmin, xmax, max_iter)

        print('ECO....')
        [bestfit2, fitness2, bestsol2, Time2] = ECO(initsol, fname, xmin, xmax, max_iter)

        print('FOA....')
        [bestfit3, fitness3, bestsol3, Time3] = FOA(initsol, fname, xmin, xmax, max_iter)

        print('CPOA....')
        [bestfit4, fitness4, bestsol4, Time4] = CPOA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_'+str(n+1)+'.npy', BestSol)

# Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Image = np.load('Preprocessed_Images_'+str(n+1)+'.npy', allow_pickle=True)
        Target = np.load('GT_'+str(n+1)+'.npy', allow_pickle=True)
        Bestsol = np.load('BestSol_'+str(n+1)+'.npy', allow_pickle=True)
        EVAL = []
        Seg = []
        Eval = np.zeros((10, 16))
        for m in range(len(Bestsol)):
            sol = Bestsol[m, :]
            Eval[m, :], SegImage = Model_DA_ViT_UNetPP(Image, Target)
        Eval[5, :], pred1 = Model_Unet(Image, Target)
        Eval[6, :], pred2 = Model_ResUNet(Image, Target)
        Eval[7, :], pred3 = Model_DD_Attention_ResUNet(Image, Target)
        Eval[8, :], pred4 = Model_DA_ViT_UNetPP(Image, Target)
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
        np.save('Unet_'+str(n+1)+'.npy', pred1)
        np.save('Resunet_'+str(n+1)+'.npy', pred2)
        np.save('DD_Attention_ResUNet_'+str(n+1)+'.npy', pred3)
        np.save('DA_ViT_UNetPP_'+str(n+1)+'.npy', pred4)
        np.save('Proposed_'+str(n+1)+'.npy', SegImage)
        np.save('Eval_all_Segmentation.npy.npy', EVAL)


# Optimization for Classification
an = 0
if an == 1:
    Breast_cancer = np.load('Proposed_1.npy', allow_pickle=True)
    Lung_Cancer = np.load('Proposed_2.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Breast_cancer = Breast_cancer
    Global_Vars.Lung_Cancer = Lung_Cancer
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Learning Rate and Steps per epoch
    xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = objective_function
    max_iter = 50

    # print('SAA....')
    # [bestfit1, fitness1, bestsol1, Time1] = SAA(initsol, fname, xmin, xmax, max_iter)
    #
    # print('ECO....')
    # [bestfit2, fitness2, bestsol2, Time2] = ECO(initsol, fname, xmin, xmax, max_iter)
    #
    # print('FOA....')
    # [bestfit3, fitness3, bestsol3, Time3] = FOA(initsol, fname, xmin, xmax, max_iter)
    #
    # print('CPOA....')
    # [bestfit4, fitness4, bestsol4, Time4] = CPOA(initsol, fname, xmin, xmax, max_iter)
    #
    # print('PROPOSED....')
    # [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)
    #
    # BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    BestSol = [initsol[0, :], initsol[1, :], initsol[2, :], initsol[3, :], initsol[4, :]]

    np.save('BestSolcls.npy', BestSol)



# Classification
an = 0
if an == 1:
    Breast_cancer = np.load('Proposed_1.npy', allow_pickle=True)
    Lung_cancer = np.load('Proposed_2.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    BestSol = np.load('BestSolcls.npy', allow_pickle=True)
    Data = Breast_cancer
    K = 5
    Per = 1 / 5
    Perc = round(Data.shape[0] * Per)
    Fold = []
    for i in range(K):
        Eval = np.zeros((10, 25))
        for j in range(5):
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Test_Data = Data[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Data.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Data[train_index, :]
            Train_Target = Target[train_index, :]
            Eval[j, :], pred = Model_ERMSC_ConvNeXtV2(Breast_cancer, Lung_cancer, Target, sol)  # With optimization
        Eval[5, :], pred1 = Model_FENN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred2 = Model_DCNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred3 = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred4 = Model_ERMSC_ConvNeXtV2(Train_Data, Train_Target, Test_Data,
                                                   Test_Target)  # Without optimization
        Eval[9, :] = Eval[4, :]
        Fold.append(Eval)
    np.save('Eval_all.npy', np.asarray(Fold))  # Save Eval all



plotConvResults()
Plot_Results()
Plot_ROC_Curve()
Table()
plot_seg_results()
Image_Results()
Sample_Images()