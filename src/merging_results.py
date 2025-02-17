import os
import shutil
import numpy as np
import pandas as pd
import pickle
from visualization import *

src_training_1 = r"results\trainings_archive\20241124_num_epoch_100_frac_100_test_no_standardization"
src_training_2 = r"results\trainings_archive\20241126_num_epoch_200_frac_100_no_stand_with_non_lin"
src_results = "results/trainings_archive/20241127_num_epoch_200_frac_100_no_standardization"
class_labels = ['bare', 'terrace', 'spontaneous', 'extensive', 'lawn', 'intensive']

def merge_logs(new_metrics_train:pd.DataFrame, new_metrics_val:pd.DataFrame, best_metrics:dict, best_epochs:dict, new_conf_mat:dict, src_folder):
    metrics_to_caps = {'OA':'OA', 'f1': 'F1', 'f2':'F2', 'f05': 'F05', 'precision':'Precision', 'recall':'Recall'}
    try:
        df_best_results = pd.DataFrame({
            'metric': metrics_to_caps.values(),
            'best_val': [best_metrics[metric] for metric in metrics_to_caps.keys()],
            'epoch': [best_epochs[metric] for metric in metrics_to_caps.keys()]
        })
        df_best_results.to_csv(os.path.join(src_folder, 'best_results.csv'), sep=';', index=False)
    except:
        print('Did not manage to merge the best_results')

    new_metrics_train.to_csv(os.path.join(src_folder, 'metrics_train.csv'), sep=';', index=False)
    new_metrics_val.to_csv(os.path.join(src_folder, 'metrics_val.csv'), sep=';', index=False)

    try:
        with open(os.path.join(src_folder, 'confusion_matrices.pickle'), 'wb') as file:
            pickle.dump(new_conf_mat, file)
    except:
        print("Did not manage to save the confusion matrices.")

    try:
        with open(os.path.join(src_training_1 + "/logs", "samp_logs.pickle"), 'rb') as inputfile:
            samp_logs1 = pickle.load(inputfile)
        with open(os.path.join(src_training_2 + "/logs", "samp_logs.pickle"), 'rb') as inputfile:
            samp_logs2 = pickle.load(inputfile)
        samp_logs1.update(samp_logs2)
        with open(os.path.join(src_folder, 'samp_logs.pickle'), 'wb') as file:
            pickle.dump(samp_logs1, file)
    except:
        print("Did not manage to merge the samp_logs.")

    try:
        shutil.copyfile(os.path.join(src_training_2 + '/logs', 'hyperparameters.yaml'), os.path.join(src_folder, 'hyperparameters.yaml'))
    except:
        print("Did not manage to copy the hyperparameters file.")

def merge_images(data_for_log_train, best_metrics, best_epochs, dict_conf_mats, src_folder):
    # save training_logs
    try:
        show_log_train(
            data=data_for_log_train, 
            target_src=os.path.join(src_folder, 'acc_loss_evolution.png'),
            do_save=True,
            do_show=False,
            )
    except:
        print("Did not manage to save training log image.")
    
    # save confusion matrices
    try:
        for metric in best_metrics.keys():   
                pred = dict_conf_mats[best_epochs[metric]]['pred']['val']
                target = dict_conf_mats[best_epochs[metric]]['target']['val']
                show_confusion_matrix(target_src=os.path.join(src_folder,f"confusion_matrix_epoch={best_epochs[metric]}_{metric}={round(best_metrics[metric], 2)}.png"),
                                    y_pred=pred,
                                    y_true=target, 
                                    class_labels=class_labels,
                                    title=f"Confusion Matrix - epoch={best_epochs[metric]} - {metric}={round(best_metrics[metric], 2)}",
                                    do_save=True,
                                    do_show=False
                                    )
    except:
        print("Did not manage to save the confusion matrix images.")
    

def merge_models(best_epochs, metrics1, metrics2, src_folder):
    try:
        metrics_to_caps = {'OA':'OA', 'f1': 'F1', 'f2':'F2', 'f05': 'F05', 'precision':'Precision', 'recall':'Recall'}
        for metric, epoch in best_epochs.items():
            if epoch in list(metrics1.epoch.values):
                shutil.copy(os.path.join(src_training_1 + '/models', f'model_{metrics_to_caps[metric]}.tar'), os.path.join(src_folder, f'model_{metrics_to_caps[metric]}.tar'))
            elif epoch in list(metrics2.epoch.values):
                shutil.copy(os.path.join(src_training_2 + '/models', f'model_{metrics_to_caps[metric]}.tar'), os.path.join(src_folder, f'model_{metrics_to_caps[metric]}.tar'))
            else:
                raise ValueError(f"Best epoch of metric {metric} is not matching any training!")
        shutil.copy(os.path.join(src_training_2 + '/models', 'model_last.tar'), os.path.join(src_folder, 'model_last.tar'))
    except:
        print("Did not manage to merge the models.")


def merging():
    # creating new architecture:
    assert os.path.exists('/'.join(src_results.split('/')[:-1]))
    assert os.path.exists(src_training_1)
    assert os.path.exists(src_training_2)

    if os.path.exists(src_results):
        shutil.rmtree(src_results)
    dict_dirs = {
        'root': src_results,
        'logs': src_results + '/logs',
        'images': src_results + '/images',
        'models': src_results + '/models',
        }
    for dir in dict_dirs.values():
        os.mkdir(dir)

    # merging files for training logs
    try:
        metrics_train_1 = pd.read_csv(os.path.join(src_training_1 + "/logs", "metrics_train.csv"), sep=';')
        metrics_val_1 = pd.read_csv(os.path.join(src_training_1 + "/logs", "metrics_val.csv"), sep=';')
        metrics_train_2 = pd.read_csv(os.path.join(src_training_2 + "/logs", "metrics_train.csv"), sep=';')
        metrics_val_2 = pd.read_csv(os.path.join(src_training_2 + "/logs", "metrics_val.csv"), sep=';')

        assert(set(metrics_train_1.epoch.values).intersection(set(metrics_train_2.epoch.values)) == set([]))
        assert metrics_train_1.epoch.values[-1] < metrics_train_2.epoch.values[0]
        new_metrics_train = pd.concat([metrics_train_1, metrics_train_2])
        new_metrics_val = pd.concat([metrics_val_1, metrics_val_2])

        data_for_log_train = {
            'epoch': new_metrics_train.epoch,
            'train_acc': new_metrics_train.OA,
            'val_acc': new_metrics_val.OA,
            'train_loss': new_metrics_train.loss,
            'val_loss': new_metrics_val.loss,
        }
    except:
        print("Did not manage to merge the logging files. Process aborted..")
        quit()

    # merging files for confusion matrices
    metrics = ['OA', 'f1', 'f2', 'f05', 'precision', 'recall']
    best_epochs = {metric: 0 for metric in metrics}
    best_metrics = {metric: 0 for metric in metrics}

    for metric in metrics:
         id_max = new_metrics_val[metric].idxmax()
         best_epochs[metric] = new_metrics_val.iloc[id_max]['epoch'].astype(int)
         best_metrics[metric] = new_metrics_val.iloc[id_max][metric]
        
    # _load confusion matrices
    confmat1 = {}
    try:
        with open(os.path.join(src_training_1 + "/logs", "confusion_matrices.pickle"),'rb') as dict_file:
            confmat1 = pickle.load(dict_file)         
        with open(os.path.join(src_training_2 + "/logs", "confusion_matrices.pickle"),'rb') as dict_file:
            confmat2 = pickle.load(dict_file)
        confmat1.update(confmat2)
    except:
        print('Did not managed to merge the confusion matrices files.')
    
    # creating images
    merge_images(data_for_log_train, best_metrics, best_epochs, confmat1, dict_dirs['images'])

    # creating logs
    merge_logs(new_metrics_train, new_metrics_val, best_metrics, best_epochs, confmat1, dict_dirs['logs'])

    # creating models
    merge_models(best_epochs, metrics_train_1, metrics_train_2, dict_dirs['models'])

    print("Process done.")


if __name__ == '__main__':
    merging()