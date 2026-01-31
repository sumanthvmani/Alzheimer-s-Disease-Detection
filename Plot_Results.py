import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
from itertools import cycle
from sklearn.metrics import roc_curve, confusion_matrix


def Plot_Results():
    # New color palette and new markers
    color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Updated colors
    bar_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Same colors for bar plot
    markers = ['*', 'H', 'v', 'X', 'P']  # New markers for each line
    # Load evaluation data
    for a in range(2):
        Eval = np.load('Eval_KFold.npy', allow_pickle=True)[a]

        # Metrics list
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score',
                 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'MCC']

        # Batch sizes and indices for the terms to plot
        learn = [1, 2, 3, 4, 5]
        Graph_Term = [0, 2, 3, 4, 7, 8, 11, 18]

        for j in range(len(Graph_Term)):
            # Initialize graph array
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))

            # Populate Graph array with evaluation data
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    if Graph_Term[j] == 18:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100

            # Line Plot
            plt.figure(figsize=(10, 6))
            for idx, (color, marker) in enumerate(zip(color_palette, markers)):
                plt.plot(learn, Graph[:, idx], color=color, linewidth=4, marker=marker,
                         markerfacecolor='white', markersize=8,
                         label=["BES-ASEA-ConvTNet", "BMO-ASEA-ConvTNet", "MOA-ASEA-ConvTNet",
                                "FFO-ASEA-ConvTNet", "RCP-FFO-ASEA-ConvTNet"][idx])
            plt.xticks(learn, ['1', '2', '3', '4', '5'], fontsize=10)
            plt.xlabel('KFOLD', fontsize=12)
            plt.ylabel(Terms[Graph_Term[j]], fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path1 = f"./Results/Dataset_{a + 1}_{Terms[Graph_Term[j]]}_Line.png"
            plt.savefig(path1)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            # Bar Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            X = np.arange(5)
            for idx, color in enumerate(bar_palette):
                ax.bar(X + idx * 0.15, Graph[:, idx + 5], color=color, edgecolor='k', width=0.15,
                       label=["RF", "XGBOOST", "CNN", "SEA-ConvTNet ", "RCP-FFO-ASEA-ConvTNet"][idx])
            ax.set_xticks(X + 0.3)
            ax.set_xticklabels(['1', '2', '3', '4', '5'], rotation=7)
            ax.set_xlabel('KFOLD', fontsize=12)
            ax.set_ylabel(Terms[Graph_Term[j]], fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path2 = f"./Results/Dataset_{a + 1}_{Terms[Graph_Term[j]]}_Bar.png"
            plt.savefig(path2)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def Plot_table():
    for a in range(2):
        Eval = np.load('Eval_Activation.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'FOR',
                 'PT',
                 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'MCC']

        Algorithm = ['TERMS', "BES-ASEA-ConvTNet", "BMO-ASEA-ConvTNet", "MOA-ASEA-ConvTNet",
                     "FFO-ASEA-ConvTNet", "RCP-FFO-ASEA-ConvTNet"]
        Classifier = ['TERMS', "RF", "XGBOOST", "CNN", "SEA-ConvTNet ", "RCP-FFO-ASEA-ConvTNet"]
        value = Eval[:, :, 4:]
        value[:, :, :-1] = value[:, :, :-1] * 100

        Graph_Term = [0, 11]
        for a in range(len(Graph_Term)):
            variation = ['Linear', 'Relu', 'Tanh', 'Sigmoid', 'Softmax']
            Table = PrettyTable()
            Table.add_column('Activation Function /Algorithm', variation[0:])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term[a]])
            print(
                '----------------------------------------------------Dataset_' + str(a + 1) + ' Algorithm Comparison -',
                Terms[Graph_Term[a]],
                '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column('Activation Function /Classifier', variation[0:])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j + 5, Graph_Term[a]])
            print('---------------------------------------------------Dataset_' + str(a + 1) + ' Method Comparison -',
                  Terms[Graph_Term[a]],
                  '--------------------------------------------------')
            print(Table)


def Confusion_matrix():
    for a in range(2):
        Actual = np.load(f'Actual_{a + 1}.npy', allow_pickle=True)
        Predict = np.load(f'Predict_{a + 1}.npy', allow_pickle=True)
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[4]).argmax(axis=1), np.asarray(Predict[4]).argmax(axis=1))
        sn.heatmap(cm, annot=True, fmt='g',
                   ax=ax).set(title=f'Confusion Matrix {a + 1}')
        path = f"./Results/Dataset_{a + 1}_Confusion.png"
        plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def Plot_ROC():
    for a in range(2):
        lw = 2
        cls = ["RF", "XGBOOST", "CNN", "SEA-ConvTNet ", "RCP-FFO-ASEA-ConvTNet"]
        colors1 = cycle(["plum", "red", "palegreen", "chocolate", "hotpink", "navy", ])
        colors2 = cycle(["hotpink", "plum", "chocolate", "navy", "red", "palegreen", "violet", "red"])
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[a][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[a][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())

            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label="{0}".format(cls[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = f"./Results/Dataset_{a + 1}__roc_.png"
        plt.savefig(path1)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def Plot_Fitness():
    for a in range(2):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ["BES-ASEA-ConvTNet", "BMO-ASEA-ConvTNet", "MOA-ASEA-ConvTNet",
                     "FFO-ASEA-ConvTNet", "RCP-FFO-ASEA-ConvTNet"]
        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('--------------------------------------------------Dataset_' + str(
            a + 1) + ' Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
                 label="BES-ASEA-ConvTNet")
        plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
                 label="BMO-ASEA-ConvTNet")
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
                 label="MOA-ASEA-ConvTNet")
        plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
                 label="FFO-ASEA-ConvTNet")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
                 label="RCP-FFO-ASEA-ConvTNet")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = f"./Results/Dataset_{a + 1}_conv.png"
        plt.savefig(path1)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def new_plot():
    for a in range(2):
        Eval = np.load('Eval_Batch.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
                 'NPV', 'FDR', 'F1-Score', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM',
                 'MK', 'LR+', 'LR-', 'MCC']

        Graph_Term = [0, 1, 4, 6, 9, 10]

        # Display labels for x-axis (not actual bar positions)
        x_tick_labels = [4, 8, 16, 32, 64]
        x_positions = [1, 2, 3, 4, 5]  # evenly spaced bars

        for a in range(len(Graph_Term)):
            Ev = Eval[:, 4, Graph_Term[a] + 4] * 100
            percentages = Ev.tolist()

            bar_color = '#0D4C75'
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(x_positions, percentages, color=bar_color, width=0.6)

            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                        f'{("{:.3f}".format(height))}%', ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

            # Dynamic Y-axis
            y_max = max(percentages)
            ax.set_ylim(0, y_max + 5)

            # Use evenly spaced x-positions but custom tick labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel('Batch Size', fontsize=12)
            ax.set_ylabel(Terms[Graph_Term[a]], fontsize=12)

            # Custom axis
            x_min = 0.5
            x_max = 5.5
            ax.spines['left'].set_position(('data', x_min))
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Background
            fig.patch.set_facecolor('#D3F3E0')
            ax.set_facecolor('#D3F3E0')
            plt.legend(['RCP-FFO-ASEA-ConvTNet'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True,
                       shadow=True)

            plt.tight_layout()
            path2 = f"./Results/New_{Terms[Graph_Term[a]]}_Bar.png"
            plt.savefig(path2)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


if __name__ == '__main__':
    Plot_Results()
    Plot_table()
    Plot_ROC()
    Plot_Fitness()
    new_plot()
