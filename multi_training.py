import os
import platform
import subprocess
import pandas as pd
import numpy as np
import time
from datetime import date
from src.visualization import show_grid_search, show_multi_combinations, show_one_var_combinatory, show_one_var_combinatory_all_metrics

params = None
# ===========================================
# ====== SETTING UP THE MULTI-TRAINING ======
# ===========================================
"""
In this section, you can prepare the list of hyperparmeters on-which to make the trainings.
You first need to define the mode:
    - "combinatory" is for looping on every possible combinations of hyperparameters
    - "combination" is for training on a list of different combinations of hyperparameters

Then, set the variable `do_preprocess` to True if some of the parameters are about the preprocessing.

Finally, you can create the instantiate the variable `params`. Its format will depend on the mode:
    - if "combinatory", the params needs to be a dictionnary containing the path to the parameter 
      in the .yaml file as key and the list of values to test as values.
      e.g.:
      params = {
        "training.parameters.batch_size": [6,8,10],
        "training.parameters.data_frac": [.05, .1],
      }

    - if "combination", the params needs to be a list of dictionnaries where each dictionnary contains
      the path to the parameter in the .yaml file as key and the specific value to test as values.
      e.g.:
      params = [
        {
            "preprocessing.processes.do_mask": False,
            "preprocessing.processes.do_smooth_mask": False,
        },
        {
            "preprocessing.processes.do_mask": True,
            "preprocessing.processes.do_smooth_mask": False,
        },
        {
            "preprocessing.processes.do_mask": True,
            "preprocessing.processes.do_smooth_mask": True,
        },
      ]
"""

mode = "combination"  # choose in {'combination', 'combinatory'}

do_preprocess = True

params =[
   {
       'preprocessing.processes.do_da_rotation': True,
       'preprocessing.processes.do_da_flipping': True,
   },
   {
       'preprocessing.processes.do_da_rotation': True,
       'preprocessing.processes.do_da_flipping': False,
   },
   {
       'preprocessing.processes.do_da_rotation': False,
       'preprocessing.processes.do_da_flipping': False,
   },
]  



# ====== END OF SETTING UP REGION ===========
# === don't modify the following parts ======
# ===========================================


def multi_training(mode, do_preprocess, params):
    """
    This function automates running multiple training sessions based on different parameter configurations. It creates directories, runs training in either combinatory or combination mode, collects results, and generates summary CSV files and visualizations.
    Args:
        - mode (str): Specifies the mode of operation ("combinatory" or "combination").
        - do_preprocess (bool): A flag indicating whether to perform preprocessing before training.
        - params (dict): A dictionary of parameters or combinations of parameters for training.

    Returns:
        - None
    """
    # security
    if mode == "combinatory":
        for values in params.values():
            assert isinstance(values, list)
            assert len(values) >= 1
    elif mode == "combination":
        for combin in params:
            assert len(combin) >= 1
    else:
        raise ValueError("Unknown value for the mode of multi-training")
    
    time_begining = time.time() # starting timer

    # get list of variables of interest
    lst_vars = []
    if mode == "combinatory":
        lst_vars = [x.split('.')[-1] for x in params.keys()]
    else:
        lst_vars = [x.split('.')[-1] for x in params[0].keys()]

    # create architecture
    #   _create folder name
    folder_name = "MULTI_" + date.today().strftime("%Y%m%d")
    for var in lst_vars:
        folder_name += "_" + var
    new_folder_name = folder_name
    i = 1
    while os.path.exists("results/trainings/" + new_folder_name):
        new_folder_name = folder_name + "_" + str(i)
        i += 1
    folder_name = new_folder_name + "/"

    #  _create other folders in folder_name
    dirs = {}
    dirs['resroot_dir'] = os.path.join("results/trainings/", folder_name)
    dirs['single_trainings_dir'] = os.path.join(dirs['resroot_dir'], 'single_trainings/')
    dirs['best_results_dir'] = os.path.join(dirs['resroot_dir'], "best_results/")
    dirs['images_dir'] = os.path.join(dirs['resroot_dir'], "images/")
    for dir in dirs.values():
        os.mkdir(dir)
    
    # run multi-training
    # prepare the list to run

    if mode == "combinatory":
        if platform.system() == 'Windows':
            lst_command = ["./.venv/Scripts/python",".multi_training_src.py", "--multirun"]
        else:
            lst_command = ["./.venv/bin/python",".multi_training_src.py", "--multirun"]
        for param, values in params.items():
            str_param = param + "=" + str(values[0])
            for x in range(1, len(values)):
                str_param += "," + str(values[x])
            lst_command.append(str_param)
        lst_command.append("do_preprocess=" + str(do_preprocess))
        lst_command.append("training.outputs.res_dir=" + dirs['single_trainings_dir'])
        lst_command.append("training.training_mode=multi")
        lst_command.append("sweeping_var=" + str(list(params.keys())))
        subprocess.run(lst_command)
    elif mode == "combination":
        for combination in params:
            if platform.system() == 'Windows':
                lst_command = ["./.venv/Scripts/python",".multi_training_src.py"]
            else:
                lst_command = ["./.venv/bin/python",".multi_training_src.py"]
            for param, values in combination.items():
                str_param = param + "=" + str(values)
                lst_command.append(str_param)
            lst_command.append("do_preprocess=" + str(do_preprocess))
            lst_command.append("training.outputs.res_dir=" + dirs['single_trainings_dir'])
            lst_command.append("training.training_mode=multi")
            lst_command.append("sweeping_var=" + str(list(combination.keys())))
            subprocess.run(lst_command)
    # run the multi-training
   # subprocess.run(lst_command)

    # regroup results
    dict_results = {}
    for x in sorted(os.listdir(dirs['single_trainings_dir'])):
        best_res_dir = os.path.join(os.path.join(dirs['single_trainings_dir'], x), "logs", "best_results.csv")
        dict_results[x] = pd.read_csv(best_res_dir, sep=';')
    
    best_results = {}
    lst_metrics = dict_results[list(dict_results.keys())[0]].metric.unique()
    for metric in lst_metrics:
        lst_results = []
        lst_columns = ["best_val","epoch"]
        for var in lst_vars:
            lst_columns.append(var)
        for name, res in dict_results.items():
            single_res = []
            single_res.append(res.loc[res['metric'] == metric, 'best_val'].values[0])
            single_res.append(res.loc[res['metric'] == metric, 'epoch'].values[0])
            for var in lst_vars:
                single_res.append(name.split(var+"=")[1].split('_')[0])

            lst_results.append(single_res)
        # save csv for metric
        best_results[metric] = pd.DataFrame(lst_results, columns=lst_columns)
        for var in lst_vars:    # converts the numerical variables to corresponding type
            try:
                best_results[metric][var] = best_results[metric][var].astype(int)
            except ValueError:
                try:
                    best_results[metric][var] = best_results[metric][var].astype(float)
                except ValueError:
                    continue
        
        best_results[metric] = best_results[metric].sort_values(lst_vars).reset_index(drop=True)    # sorts the data with respect to the variables
        best_results[metric].to_csv(os.path.join(dirs['best_results_dir'],f'best_{metric}.csv'), sep=';', index=False)

    # save figures
    if mode == 'combination':
        for metric in lst_metrics:
            show_multi_combinations(
                target_src=os.path.join(dirs['images_dir'], f"combination_results_{metric}.png"),
                data=best_results[metric],
                title=f"combination results - {metric}",
                do_save=True,
                do_show=False,
                )
    else:
        if len(params) == 1:
            print("\nsaving 1D results...")
            data_all_metrics = pd.DataFrame(columns=['best_val', lst_vars[0], 'metric'])
            for metric in lst_metrics:
                best_results[metric]['metric'] = metric
                data_all_metrics = pd.concat([data_all_metrics.dropna(axis=1, how='all'), best_results[metric].dropna(axis=1, how='all')], ignore_index=True)
                show_one_var_combinatory(
                    target_src=os.path.join(dirs['images_dir'], f"one_var_results_{metric}.png"),
                    data=best_results[metric],
                    title=f"one var - results - {metric}",
                    do_save=True,
                    do_show=False,
                    )
            show_one_var_combinatory_all_metrics(
                target_src=os.path.join(dirs['images_dir'], f"one_var_results_all_metrics.png"),
                data=data_all_metrics, 
                title=f"{lst_vars[0]} - results", 
                tested_var=lst_vars[0], 
                do_save=True, 
                do_show=False,
                )
            
        elif len(params) == 2:
            print("\nsaving 2D results...")
            for metric in lst_metrics:
                show_grid_search(
                    target_src=os.path.join(dirs['images_dir'], f"grid_search_{metric}.png"),
                    data=best_results[metric],
                    title=f"grid search results - {metric}",
                    do_save=True,
                    do_show=False,
                    )

    # print time to process
    time_elapsed = time.time() - time_begining
    n_hours = int(time_elapsed / 3600)
    n_min = int((time_elapsed % 3600) / 60)
    n_sec = int(time_elapsed - n_hours * 3600 - n_min * 60)
    print("\n==============")
    print(f'Multi-training complete in {n_hours}:{n_min}:{n_sec}')
    print("==============")


if __name__ == "__main__":
    multi_training(mode, do_preprocess, params)

    """lst_vars = ['batch_size', 'data_frac']
    dict_results = {}
    dir_sgl_trainings = "./results/trainings/MULTI_20241008_batch_size_data_frac_2/single_trainings"
    for x in sorted(os.listdir(dir_sgl_trainings)):
        print(x)
        best_res_dir = os.path.join(os.path.join(dir_sgl_trainings, x), "logs", "best_results.csv")
        dict_results[x] = pd.read_csv(best_res_dir, sep=';')
    
    print(dict_results)
    
    best_results = {}
    lst_metrics = dict_results[list(dict_results.keys())[0]].metric.unique()
    for metric in lst_metrics:
        lst_results = []
        lst_columns = ["best_val","epoch"]
        for var in lst_vars:
            lst_columns.append(var)
        for name, res in dict_results.items():
            single_res = []
            single_res.append(res.loc[res['metric'] == metric, 'best_val'].values[0])
            single_res.append(res.loc[res['metric'] == metric, 'epoch'].values[0])
            for var in lst_vars:
                single_res.append(name.split(var+"=")[1].split('_')[0])
            lst_results.append(single_res)

        print

        best_results[metric] = pd.DataFrame(lst_results, columns=lst_columns)
        for var in lst_vars:
            try:
                best_results[metric][var] = best_results[metric][var].astype(int)
            except ValueError:
                try:
                    best_results[metric][var] = best_results[metric][var].astype(float)
                except ValueError:
                    continue
        best_results[metric] = best_results[metric].sort_values(lst_vars).reset_index(drop=True)
        print(best_results[metric])
    #df = pd.read_csv("./results/trainings/MULTI_20241008_batch_size_data_frac_2/best_results/best_F1.csv", sep=';')
    #print(df.sort_values(['batch_size', 'data_frac']).reset_index(drop=True))"""
