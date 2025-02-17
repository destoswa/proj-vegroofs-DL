import sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, recall_score, precision_score
import seaborn as sns
import pandas as pd
import pickle
import warnings
from scipy.ndimage import uniform_filter1d
if __name__=='__main__':
    sys.path.insert(0, 'D:\GitHubProjects\STDL_Classifier_uncertainty')
from src.uncertainty_utils import brier_score_multiclass, expected_calibration_error, average_prediction_entropy, multiclass_nll, uncertainty_aware_accuracy
import os

# LOADING DATA


def show_log_train(data, target_src, do_save=True, do_show=False):
    """
    Plot accuracy and loss curves for training and evaluation data.

    Args:
        - data (dict): Dictionary with keys 'train_acc', 'train_loss', 'val_acc', 'val_loss', and 'epoch' representing training and validation metrics.
        - target_src (str): File path where the plot will be saved.
        - do_save (bool, optional): If True, save the plot. Default is True.
        - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
        - None
    """
    #data = pd.read_csv(data_src, delimiter=';')
    warnings.filterwarnings('ignore')
    ls_train_acc = data['train_acc'].to_list()
    ls_train_loss = data['train_loss'].to_list()
    ls_val_acc = data['val_acc'].to_list()
    #ls_test_class_acc = data['test_class_acc'].to_list()
    ls_val_loss = data['val_loss'].to_list()
    ls_epochs = data['epoch'].to_list()

    # Plot results
    fig, axs = plt.subplots(2, 1, sharex=True)

    # splines
    """spline_train_loss = UnivariateSpline(ls_epochs, ls_train_loss, s=1)
    spline_val_loss = UnivariateSpline(ls_epochs, ls_val_loss, s=1)
    spline_train_acc = UnivariateSpline(ls_epochs, ls_train_acc, s=1)
    spline_val_acc = UnivariateSpline(ls_epochs, ls_val_acc, s=1)
    x_smooth = np.linspace(0, ls_epochs[-1], 1000)
    y_smooth_train_loss = spline_train_loss(x_smooth) 
    y_smooth_val_loss = spline_val_loss(x_smooth) 
    y_smooth_train_acc = spline_train_acc(x_smooth) 
    y_smooth_val_acc = spline_val_acc(x_smooth) """

    # moving window
    n = 5
    num_rep = 5
    y_smooth_train_loss = uniform_filter1d(ls_train_loss, size=n)
    y_smooth_val_loss = uniform_filter1d(ls_val_loss, size=n)
    y_smooth_train_acc = uniform_filter1d(ls_train_acc, size=n)
    y_smooth_val_acc = uniform_filter1d(ls_val_acc, size=n)

    for i in range(num_rep - 1):
        y_smooth_train_loss = uniform_filter1d(y_smooth_train_loss, size=n)
        y_smooth_val_loss = uniform_filter1d(y_smooth_val_loss, size=n)
        y_smooth_train_acc = uniform_filter1d(y_smooth_train_acc, size=n)
        y_smooth_val_acc = uniform_filter1d(y_smooth_val_acc, size=n)

    # plot accuracies
    axs[0].plot(ls_epochs, ls_train_acc, label='train', alpha=0.3)
    axs[0].plot(ls_epochs, ls_val_acc, label='eval', alpha=0.3)
    axs[0].plot(ls_epochs, y_smooth_train_acc, color="#1f77b4", label='_nolegend_', linewidth=1.5)
    axs[0].plot(ls_epochs, y_smooth_val_acc, color="#ff7f0e", label='_nolegend_', linewidth=1.5)
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Accuracy value [-]')
    axs[0].set_ylim(None, 1.0)
    axs[0].legend(loc='upper left')

    # plot losses
    axs[1].plot(ls_epochs, ls_train_loss, label='train', alpha=0.3)
    axs[1].plot(ls_epochs, ls_val_loss, label='eval', alpha=0.3)
    axs[1].plot(ls_epochs, y_smooth_train_loss, color="#1f77b4", label='_nolegend_', linewidth=1.5)
    axs[1].plot(ls_epochs, y_smooth_val_loss, color="#ff7f0e", label='_nolegend_', linewidth=1.5)
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss value [-]')
    axs[1].legend()

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_precision_recall(data, target_src, do_save=True, do_show=False):
    """
    Plot precision and recall curves over training epochs.

    Args:
    - data (dict): Dictionary with 'precision', 'recall', and 'epoch' as keys.
    - target_src (str): File path where the plot will be saved.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    warnings.filterwarnings('ignore')
    ls_recall = data['recall'].to_list()
    ls_precision = data['precision'].to_list()
    epochs = data['epoch'].to_list()

    # Plot results
    fig = plt.figure()
    plt.plot(epochs, ls_precision, label='precision')
    plt.plot(epochs, ls_recall, label='recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch [-]')
    plt.ylabel('Metric [-]')
    plt.legend(loc="lower right")

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_metrics_evolution(data, target_src, title, do_save=True, do_show=False):
    """
    Plot the evolution of multiple metrics over training epochs.

    Args:
    - data (DataFrame): DataFrame with metrics and 'epoch' column.
    - target_src (str): File path where the plot will be saved.
    - title (str): Title of the plot.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    warnings.filterwarnings('ignore')
    epochs = data['epoch'].to_list()
    lst_metrics = data.drop(['epoch'], axis=1).columns.tolist()
    fig = plt.figure()
    for metric in lst_metrics:
        plt.plot(epochs, data[metric], label=metric)
    plt.grid()
    plt.title(title)
    plt.xlabel('Epoch [-]')
    plt.ylabel('Value [-]')
    plt.legend(loc="lower right")

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_confusion_matrix(y_pred, y_true, target_src,class_labels, title="", do_save=True, do_show=False):
    """
    Plot confusion matrix and associated metrics.

    Args:
    - y_pred (list): List of predicted labels.
    - y_true (list): List of true labels.
    - target_src (str): File path where the plot will be saved.
    - class_labels (list): List of class names corresponding to the labels.
    - title (str, optional): Title of the plot. Default is an empty string.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    warnings.filterwarnings('ignore')
    n_classes = len(class_labels)

    # confusion matrix
    conf_mat = np.round(confusion_matrix(y_true, y_pred, labels=range(0, n_classes), normalize='true'),2)
    df_conf_mat = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)

    # metrics
    recall = recall_score(y_true, y_pred, labels=range(0, n_classes), average=None)
    precision = precision_score(y_true, y_pred, labels=range(0, n_classes), average=None)
    f1 = f1_score(y_true, y_pred, labels=range(0, n_classes), average=None)
    f2 = fbeta_score(y_true, y_pred, labels=range(0, n_classes), average=None, beta=2)
    f05 = fbeta_score(y_true, y_pred, labels=range(0, n_classes), average=None, beta=0.5)
    acc = confusion_matrix(y_true, y_pred, labels=range(0, n_classes)).diagonal() / confusion_matrix(y_true, y_pred, labels=range(0, n_classes)).sum(axis=1)


    df_metrics = pd.DataFrame(index=class_labels)
    df_metrics['OA'] = acc
    df_metrics['Recall'] = recall
    df_metrics['F2'] = f2
    df_metrics['F1'] = f1
    df_metrics['F05'] = f05
    df_metrics['Precision'] = precision
    df_metrics.loc["Global"] = [
        accuracy_score(y_true, y_pred),
        recall_score(y_true, y_pred, labels=range(0, n_classes), average="macro"),
        fbeta_score(y_true, y_pred, labels=range(0, n_classes), average="macro", beta=2),
        f1_score(y_true, y_pred, labels=range(0, n_classes), average="macro"),
        fbeta_score(y_true, y_pred, labels=range(0, n_classes), average="macro", beta=0.5),
        precision_score(y_true, y_pred, labels=range(0, n_classes), average="macro"),
        ]

    # plotting
    fig,axs = plt.subplots(1,2,figsize=(10,6))
    subfig_confmat = sns.heatmap(df_conf_mat, annot=True, cmap=sns.color_palette("Blues", as_cmap=True), cbar=False, ax=axs[0])
    subfig_metrics = sns.heatmap(df_metrics, annot=True, cmap=sns.color_palette("Blues", as_cmap=True), cbar_kws = dict(location="left"), ax=axs[1])
    subfig_metrics.tick_params(right=True, labelright=True, labelleft=False)
    plt.yticks(rotation='horizontal')
    axs[0].set_ylabel('True labels')
    axs[0].set_xlabel('Predicted labels')
    axs[0].set_title('Confusion matrix')
    axs[1].set_title('Metrics')
    axs[1].set_xlabel('Indexes')
    fig.suptitle(title)
    plt.tight_layout()

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_grid_search(target_src, data, title, do_save=True, do_show=False):
    """
    Plot the results of a grid search over 2 hyperparameters.

    Args:
    - target_src (str): File path where the plot will be saved.
    - data (DataFrame): DataFrame containing grid search results.
    - title (str): Title of the plot.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()
    X = data.iloc[:, -1].unique()
    Y = data.iloc[:, -2].unique()
    Z = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[i, j] = data[(data.iloc[:, -1] == x) & (data.iloc[:, -2] == y)].iloc[0, 0]

    glue = data.pivot(index=data.columns[-1], columns=data.columns[-2], values="best_val")
    sns.heatmap(glue, annot=True, cmap=sns.color_palette("Blues", as_cmap=True))
    ax.set_title(title)
    ax.set_xlabel(data.columns[-2])
    ax.set_ylabel(data.columns[-1])

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_multi_combinations(target_src, data, title, do_save=True, do_show=False):
    """
    Plot results for multiple combinations of hyperparameters.

    Args:
    - target_src (str): File path where the plot will be saved.
    - data (DataFrame): DataFrame containing combination results.
    - title (str): Title of the plot.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    warnings.filterwarnings('ignore')
    x_tickslabels = []
    for _, row in data.iterrows():
        ticklabel = ""
        for i in range(1,len(row)-1):
            #ticklabel += (f"{row.keys()[-i]} = {row[-i]}\n")
            ticklabel += (f"{row[-i]}")
        x_tickslabels.append(ticklabel)
    data = pd.concat([data,pd.DataFrame(x_tickslabels, columns=['ticklabel'])], axis=1)
    
    fig = plt.figure()
    g = sns.barplot(data, x="ticklabel", y='best_val')
    g.bar_label(g.containers[0])
    g.set(xlabel=None)
    g.set(ylabel="Metric's value [-]")
    plt.title(title)
    plt.tight_layout()

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_one_var_combinatory(target_src, data, title, do_save=True, do_show=False):
    """
    Plot a single-variable combinatory result.

    Args:
    - target_src (str): File path where the plot will be saved.
    - data (DataFrame): DataFrame containing combinatory results for one variable.
    - title (str): Title of the plot.
    - class_labels (list): List of class labels.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()
    g = sns.lineplot(data, x=data.columns[-1], y='best_val', marker='o')
    g.set(ylabel="Metric's value [-]")
    for x,y in data[[data.columns[-1], 'best_val']].values:
        t = ax.text(x,y,f'{y:.2f}')
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='red'))
    #plt.xscale('log')
    plt.title(title)
    plt.tight_layout()

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_one_var_combinatory_all_metrics(target_src, data: dict, title, tested_var, do_save=True, do_show=False):
    """
    Plot a single-variable combinatory result.

    Args:
    - target_src (str): File path where the plot will be saved.
    - data (DataFrame): DataFrame containing combinatory results for one variable.
    - title (str): Title of the plot.
    - class_labels (list): List of class labels.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    # color palette
    distinct_colors = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#FFA500",  # Orange
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#800080",  # Purple
    "#808080",  # Gray
    "#A52A2A"   # Brown
    "#FFFF00",  # Yellow
]

    warnings.filterwarnings('ignore')
    metrics_order =  ['OA', 'Recall', 'F2', 'F1', 'F05', 'Precision']
    data = data.set_index('metric').loc[metrics_order].reset_index()
    print(data)
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='metric', y='best_val', hue=tested_var, marker='s', linestyle='dotted', palette=distinct_colors, ax=ax)
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Best val [-]')
    plt.tight_layout()

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


def show_uncertainty(dict_preds_conf, target_src, do_show=True, do_save=False):
    epochs = dict_preds_conf.keys()
    dict_scores = {
        'Brier score': [],
        'ECE': [],
        'APE': [],
        'NLL': [],
        'UAA': [],
    }
    for dict in dict_preds_conf.values():
        targets = np.array(dict['val']['target'])
        preds = np.array(dict['val']['pred'])
        preds_conf = np.array(dict['val']['pred_conf'])
        
        dict_scores['Brier score'].append(brier_score_multiclass(y_true=targets, y_pred=preds_conf))
        dict_scores['ECE'].append(expected_calibration_error(y_true=targets, y_pred_lbl=preds, y_pred=preds_conf))
        dict_scores['APE'].append(average_prediction_entropy(pred_probs=preds_conf))
        dict_scores['NNL'].append(multiclass_nll(y_true=targets, y_pred=preds_conf))
        dict_scores['UAA'].append(uncertainty_aware_accuracy(predicted_labels=preds, pred_probs=preds_conf, true_labels=targets)[0])

    min_brier = min(dict_scores['Brier score'])
    min_ECE = min(dict_scores['ECE'])
    min_APE = min(dict_scores['APE'])
    min_NNL = min(dict_scores['NNL'])
    max_UAA = max(dict_scores['UAA'])
    fig, axs = plt.subplots(3, 2)
    for idx, (score_name, score_val) in enumerate(dict_scores.items()):
        num_row = idx % 3
        num_col = idx // 3
        ax = axs[num_row, num_col]
        ax.plot(epochs, score_val)
        ax.set_title(score_name)
        ax.grid()
    axs[2,1].text(0.05, 0.85, f"Min Brier score : {round(min_brier, 3)}", color='g', fontsize=8)
    axs[2,1].text(0.05, 0.65, f"Min Expected Calibration Error : {round(min_ECE, 3)}", color='g', fontsize=8)
    axs[2,1].text(0.05, 0.45, f"Min Average Prediction Entropy: {round(min_APE, 3)}", color='g', fontsize=8)
    axs[2,1].text(0.05, 0.25, f"Min Negative Log-likelihood : {round(min_NNL, 3)}", color='g', fontsize=8)
    axs[2,1].text(0.05, 0.05, f"Max Uncertainty-Aware Acc : {round(max_UAA, 3)}", color='g', fontsize=8)
    axs[2,1].set_xticks([])
    axs[2,1].set_yticks([])
    axs[2,1].set_title("Best value for each score")
    fig.suptitle("Measure of uncertainty")
    plt.tight_layout()

    if do_save:
        plt.savefig(target_src)

    if do_show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # show uncertainty results
    src = r"D:\GitHubProjects\STDL_Classifier_uncertainty\results\trainings\20250121_num_epoch_3_frac_10_test_8\logs\samp_logs.pickle"
    target_src = r"D:\GitHubProjects\STDL_Classifier_uncertainty\results\trainings\20250121_num_epoch_3_frac_10_test_8\images\uncertainty.png"
    with open(src,'rb') as file:
        preds_conf = pickle.load(file)
    
    show_uncertainty(preds_conf, target_src, True, False)
    quit()

    # load results
    conf_mat_src = "./logs/MSPP_num_epoch_50_250924/conf_mats/conf_mat_4.csv"
    metric_train_src = "./logs/MSPP_num_epoch_50_250924/metrics_train.csv"
    metric_val_src = "./results/trainings/20241010_num_epoch_10_frac_10_test/logs/metrics_val.csv"
    metric_test_prec_rec_src = './results/trainings/20241010_num_epoch_10_frac_10_test/logs/metrics_train.csv'
    
    #from scipy.ndimage import uniform_filter1d

    # show training results
    src = r"D:\GitHubProjects\STDL_Classifier\results\trainings_archive\20241128_num_epoch_100_frac_100_deep_binary_no_do"
    src_logs = src + "\logs"
    df_metrics_train = pd.read_csv(os.path.join(src_logs, 'metrics_train.csv'), sep=';')
    df_metrics_val = pd.read_csv(os.path.join(src_logs, 'metrics_val.csv'), sep=';')

    res_logs = {
        'epoch': df_metrics_train.index,
        'train_acc': df_metrics_train.OA,
        'train_loss': df_metrics_train.loss,
        'val_acc': df_metrics_val.OA,
        'val_loss': df_metrics_val.loss,
    }

    show_log_train(res_logs, os.path.join(src + "/images", 'acc_loss_evolution.png'), False, True)

    quit()


    # show one_var  
    list_images = [
        ("F1", "best_F1.csv", "one_var_results_F1.png"),
        ("F2", "best_F2.csv", "one_var_results_F2.png"),
        ("F05", "best_F05.csv", "one_var_results_F05.png"),
        ("OA", "best_OA.csv", "one_var_results_OA.png"),
        ("Precision", "best_Precision.csv", "one_var_results_Precision.png"),
        ("Recall", "best_Recall.csv", "one_var_results_Recall.png"),
    ]
    data_all_metrics = pd.DataFrame(columns=['best_val', 'rangelimit_mode', 'metric'])
    for (metric, src_file, dest_file) in  list_images:
        src = os.path.join("./results/trainings_archive/MULTI_20241110_learning_rate/best_results", src_file)
        dst = os.path.join("./results/trainings_archive/MULTI_20241110_learning_rate/images", dest_file)
        one_var_data = pd.read_csv(src, sep=';')
        df = one_var_data.copy()
        df['metric'] = metric
        data_all_metrics = pd.concat([data_all_metrics.dropna(axis=1, how='all'), df.dropna(axis=1, how='all')], ignore_index=True)
        show_one_var_combinatory(dst, one_var_data, "one var - results - " + metric, do_save=True, do_show=False)

    show_one_var_combinatory_all_metrics("./results/trainings_archive/MULTI_20241110_learning_rate/images",
                                         data_all_metrics, "learning rates - results", 'learning_rate', do_save=True, do_show=True)
    
    quit()


    # show multi_combination
    target_folder = "./results/trainings/mutli_num_epoch_100_atrous/"
    lst_metrics = ['F1', 'F2', 'F05', 'OA', 'Precision', 'Recall']
    for metric in lst_metrics:
        print(os.path.join(target_folder + 'best_results/', f"best_{metric}.csv"))
        combination_data = pd.read_csv(os.path.join(target_folder + 'best_results/', f"best_{metric}.csv"), sep=';')
        print(combination_data)
        show_multi_combinations(
            target_src=os.path.join(target_folder + 'images/', f'combination_results_{metric}.png'), 
            data= combination_data,
            title=f"combination results - {metric}", 
            do_save=True, 
            do_show=False,
            )
    quit()

    # show confusion matrix
    class_labels = ['bare', 'terrace', 'spontaneous', 'extensive', 'lawn', 'intensive']
    #class_labels = ['bare','vegetated']
    target_folder = "./results/trainings_archive/20241028_num_epoch_200_frac_100_tlm_deep_512/"
    best_epoch = 126
    #df_conv_mat = pd.read_csv(conf_mat_src, sep=';')
    with open(os.path.join(target_folder, "logs/confusions_matrices.pickle"), 'rb') as output_file:
        confusion_matrices = pickle.load(output_file)
    pred = confusion_matrices[best_epoch]['pred']['val']
    target = confusion_matrices[best_epoch]['target']['val']

    show_confusion_matrix(target_src=os.path.join(target_folder, "images/new_confusion_matrix_epoch=103_Recall=0.64.png"),
                          y_pred=pred,
                          y_true=target, 
                          class_labels=class_labels,
                          title='Performances - recall=0.69 - epoch=126',
                          do_save=False,
                          do_show=True,
                          )
    quit()

    
    # show metric evolution
    df_metrics = pd.read_csv(metric_test_prec_rec_src, sep=';')
    df_metrics = df_metrics[['epoch', 'f1','f2','f05']]

    show_metrics_evolution(df_metrics, './data/test/test_show_metrics.csv', 'test - metrics evolution', do_save=False, do_show=True)
    quit()

    

    # show grid-search
    grid_search_data = pd.read_csv('./results/trainings/MULTI_20241008_batch_size_data_frac/best_results/best_Recall.csv', sep=';')
    show_grid_search("./results/trainings/MULTI_20241008_batch_size_data_frac/images/grid_search_test.png", 
                     grid_search_data, "test grid_search", do_save=False, do_show=True)
    
    quit()


    # test show precision and recall
    df_metrics = pd.read_csv(metric_test_prec_rec_src, sep=';')
    show_precision_recall(data=df_metrics, target_src="./results/trainings/20240930_num_epoch_30_frac_1/images/precision_recall_logs.png",do_show=False, do_save=True)
    
    quit()
