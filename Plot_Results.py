import numpy as np
import pylab
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.patheffects as pe

from Image_Results import *

No_of_Dataset = 1


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'SAA-ExARMCC', 'ECO-ExARMCC', 'IFOA-ExARMCC', 'CPOA-ExARMCC', 'RHCPO-ExARMCC']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('--------------------------------------------------Dataset-', i+1,  ' - Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        fig = plt.figure(facecolor='#bdbdbd')
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        ax.yaxis.grid()
        ax.set_facecolor("#e0f3db")
        fig.canvas.manager.set_window_title('Dataset - ' + str(i + 1) + ' - Convergence Curve')
        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    cls = ['FENN', 'EDCNN', 'RAN', 'ExARMCC', 'RHCPO-ExARMCC']
    for a in range(No_of_Dataset):  # For 2 Datasets
        Actual = np.load('Target.npy', allow_pickle=True).astype('float64')
        lenper = round(Actual.shape[0] * 0.75)
        # Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset - ' + str(a + 1) + ' - ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i].astype('float64')
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_%s.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def Table():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Algorithm = ['TERMS/Epoch', 'SAA-ExARMCC', 'ECO-ExARMCC', 'IFOA-ExARMCC', 'CPOA-ExARMCC', 'RHCPO-ExARMCC']
    Classifier = ['TERMS/Epoch', 'FENN', 'EDCNN', 'RAN', 'ExARMCC', 'RHCPO-ExARMCC']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Table_Terms = [0, 3, 9, 18]
    table_terms = [Terms[i] for i in Table_Terms]
    Epoch = [50, 100, 150, 200, 250]
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Epoch)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Epoch)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plot_Results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Algorithm = ['SAA-ExARMCC', 'ECO-ExARMCC', 'IFOA-ExARMCC', 'CPOA-ExARMCC', 'RHCPO-ExARMCC']
    Classifier = ['FENN', 'EDCNN', 'RAN', 'ExARMCC', 'RHCPO-ExARMCC']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 3, 5, 8, 9, 10, 12, 13, 14]
    kFOLD = [1, 2, 3, 4, 5]
    for i in range(1):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            # ax.yaxis.grid()
            # ax.set_facecolor("#e0f3db")
            # fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Algorithm Comparison of BatchSize')
            plt.plot(kFOLD, Graph[:, 0], lw=5, color='blue',
                     path_effects=[pe.withStroke(linewidth=8, foreground='violet')], marker='h',
                     markerfacecolor='blue', markersize=5,
                     label=Algorithm[0])
            plt.plot(kFOLD, Graph[:, 1], lw=5, color='maroon',
                     path_effects=[pe.withStroke(linewidth=8, foreground='tan')], marker='h',
                     markerfacecolor='#7FFF00', markersize=5,
                     label=Algorithm[1])
            plt.plot(kFOLD, Graph[:, 2], lw=5, color='lime',
                     path_effects=[pe.withStroke(linewidth=8, foreground='orange')], marker='h',
                     markerfacecolor='#808000',
                     markersize=5,
                     label=Algorithm[2])
            plt.plot(kFOLD, Graph[:, 3], lw=5, color='deeppink',
                     path_effects=[pe.withStroke(linewidth=8, foreground='w')], marker='h', markerfacecolor='#CD1076',
                     markersize=5,
                     label=Algorithm[3])
            plt.plot(kFOLD, Graph[:, 4], lw=5, color='k',
                     path_effects=[pe.withStroke(linewidth=8, foreground='red')], marker='h', markerfacecolor='black',
                     markersize=5,
                     label=Algorithm[4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                       ncol=3, fancybox=True)
            plt.xticks(kFOLD, ('1', '2', '3', '4', '5'), fontsize=12)
            plt.xlabel('KFOLD', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            import pandas as pd
            kfol = [0, 1, 2, 3, 4]
            Graph1 = Graph[:, 5:]
            df1 = pd.DataFrame(Graph1)
            ax = df1.plot(kind='bar', legend=False, figsize=(10, 6), rot=0, width=0.8)

            bars = ax.patches
            hatches = ''.join(h * len(df1) for h in 'x/O.*')

            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            plt.legend(['FENN', 'EDCNN', 'RAN', 'ExARMCC', 'RHCPO-ExARMCC'], loc='upper center', bbox_to_anchor=(0.5, 1.10),
                       ncol=5, fancybox=True)
            plt.xticks(kfol, ('1', '2', '3', '4', '5'), fontname="Arial", fontsize=12)
            plt.xlabel('KFOLD', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            # plt.ylim([65, 95])
            path = "./Results/Dataset_%s_%s_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def plot_seg_results():
    Eval_all = np.load('Eval_all_Segmentation.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'SAA-AEDVNS-EUNet++', 'ECO-AEDVNS-EUNet++', 'IFOA-AEDVNS-EUNet++', 'CPOA-AEDVNS-EUNet++', 'RHCPO-AEDVNS-EUNet++']
    Methods = ['TERMS', 'U-Net', 'ResUnet', 'DDA-AttResUNet', 'AEDVNS-EUNet++', 'RHCPO-AEDVNS-EUNet++']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])
            ax.bar(X + 0.00, stats[i, 0, :], color='yellow', width=0.10, label=Algorithm[1])
            ax.bar(X + 0.10, stats[i, 1, :], color='c', width=0.10, label=Algorithm[2])
            ax.bar(X + 0.20, stats[i, 2, :], color='orange', width=0.10, label=Algorithm[3])
            ax.bar(X + 0.30, stats[i, 3, :], color='g', width=0.10, label=Algorithm[4])
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label=Algorithm[5])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23),
                       ncol=2, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            path = "./Results/Dataset-%s-%s_Algorithm.png" % (n+1, Terms[i - 4])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])
            ax.bar(X + 0.00, stats[i, 5, :], color='yellow', width=0.10, label=Methods[1])
            ax.bar(X + 0.10, stats[i, 6, :], color='c', width=0.10, label=Methods[2])
            ax.bar(X + 0.20, stats[i, 7, :], color='orange', width=0.10, label=Methods[3])
            ax.bar(X + 0.30, stats[i, 8, :], color='g', width=0.10, label=Methods[4])
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label=Methods[5])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            path = "./Results/Dataset-%s-%s_Method.png" % (n+1, Terms[i - 4])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_Results()
    Plot_ROC_Curve()
    Table()
    plot_seg_results()
    Image_Results()
    Sample_Images()
