# Deep learning base method for green roofs classification 

## Installation
### downolading data
A toy example of dataset is available. In order to load it, run the script `load_example_data.py` which is going to download the rasters and the AOI in the location `./data/sources/example`

### python version
This project was developped with **python3.9** version. In order to lower the risk of incompatibilities, we suggest to use the same version.

### creating virtual environment
To create the virtual environment for the project, launch a terminal from the root of the project and type:
```
python -m venv .venv
```
Once the virtual environement is created, active it by typing the following command (depending on the os):

if on Windows:
```
.venv/Scripts/activate
```

if on Mac or Linux:
```
source .venv/bin/activate
```

### installing libraries
To install the libraries, then run the following command (from the root of the project as well):
```
pip install -r ./requirements.txt
```

## architecture
The architecture of the project is as following:
```
├── config                          # config files
|  |    config.yaml
|  |    preprocessing.yaml
|  |    training.yaml
|  |    inference.yaml
|  |    bigdata.yaml
├── data                            # data to process, see readme.md inside it
├── models                          # scripts and versions about models
|  ├── inference                    # pretrained model(s) used for inference
|  |   |    model.tar
|  ├── pretrained                   # pretrained model(s) used for trainings
|  |   |    model.tar
|  |   ASPP_Classifier.py           # script of the model
|  |   model_modules.py             # script of the different modules used by the model
├── results                         # location for results
|  ├── trainings  
├── src                             # source files containing functions and classes used by the main scripts
|  |   dataset.py                   # Define the class "GreenRoofsDataset" used to load the samples
|  |   dataset_utils.py             # functions called by the GreenRoofsDataset
|  |   merging_results.py           # script used to merge matching trainings on a splitted number of epochs (e.g. training 1: epoch 0-100, training 2: epoch 101-200)
|  |   preprocess_utils.py          # functions called by the preprocessing
|  |   uncertainty_utils.py         # functions called to process the uncertainty of the model
|  |   visualization.py             # functions called to generate the different graphics
|  preprocessing.py                 # script used to transform a set of tiles and a set of polygons into a set of samples, ready to be used to train or infer
|  training.py                      # script used to train the model
|  .multi_training_src.py           # script called by multi_training.py. It should never be changed for the multi-training to work
|  multi_training.py                # script used to do multiple trainings with different configurations
|  inference.py                     # script used to do inference using the pretrained model in the corresponding folder
|  bigdata_process.py               # script used to do inference and train on big datasets through batching
|  requirements.txt                 # dependencies of the project. List of python libraries
```

 
Remark: If multi-trainings are done, the library _hydra_ will create a folder **multirun** at the root of the project. This folder can be ignored (and already is in the .gitignore).

## Documentation
### Preprocessing
The preprocessing consists in one script `./preprocess.py` driven by one configuration file `./config/preprocess.yaml`.

#### inputs
The sources used to create the dataset are:
- rasters from measurements taken by plane in the NIR,R,G,B bands
- building's footprints as geometries

#### outputs
The results from the preprocess should be a ready-to-use dataset. In order to do so, the main aim of the `preprocess.py` script is to clip the bbox of each polygons to the rasters in order to get a single sample per roof.

#### config
The pre-processing can be driven through the config file `./config/preprocessing.yaml`. For more information on the different hyper-parameters, a `README.md` file is present at the root of the folder `./config/`
The typical parameters to set up are:
- the `inputs` parameters which are the path to the ground truth (.gpkg file), the rasters root directory (the script will look at all the \*.tiff files in this directory) and the class labels list (class_labels_\{multi, binary\}.csv file)
- the `output_dir` which is the relative path and name of the target directory
- `preprocess/do_rangelimit`, `metadata/rangelimit_mode` and `metadata/rangelimit_threshold` when using 16bits rasters

#### run
- Once the config file is set, the script `./preprocessing.py` can be run.
- WARNING: If a folder with the same name already exists at the target location, it will be erased !!

### Training
Once the dataset has been created by the `preprocessing.py` script, it can be used to train a model. In order to do so, a model can be trained from scratch or from a pretrained model.

#### inputs
The sources used to train the model are:
- the dataseet created by the preprocessing
- eventually a pre-trained model saved as `model.tar` in the folder `./pretrained/`
  
#### outputs
the results of the training are going to be saved in a folder named `<date>_num_epochs_<# epochs>_frac_<frac>_<suffixe>` (e.g. _20241010\_num\_epoch\_3\_frac\_100\_test_) in `./trainings/`. For more information, look at the section [Results](#results).

#### config
The training is driven through the config file `./config/training.yaml`. For more information on the different hyper-parameters, a `README.md` file is prensent a the root of the folder `./config/`.
The typical parameters to set up are:
- the `dataset_root` parameter which is the relative path from the root of the project to the dataset to use (i.e. the dataset created by the preprocessing)
- the `outputs` parameters which are:
  - the relative path from the root to the `trainings` directory
  - the suffixe of the training
- the `do_preprocessing` parameter to first run the preprocessing before starting the training
- the `data_frac`, `train_frac` and `num_epochs` parameters to influence the training
- the `batch_size` and `num_workers` parameters to adapt as best to the machine capacities
- the `from_pretrained` to train from scratch or from a pretrained model

Moreover, the model can set for multi-class or binary-class classification. In order to do so, two hyper-parameters need to be set:
- Regarding the preprocessing, the config parameter `inputs.class_abels_dir` gives the location of the csv file that maps the class contained in `inputs.polygon_src` to the cooresponding categories used to train the model.
- Regarding the training, the config parameter `model.model_type` allow to set the model into the correct configuration for processing the coorresponding number of classes
- 
#### run
- Once the config file is set, the script `./training.py` can be run.

### Multi-training

#### inputs
The multi-training acts as an overlay on training. Therefore, the inputs are going to be the same as the ones for training a model.
However, informations regarding the variables to grid-search need to be provided directly in the script (see bellow in `config`).

#### outputs
The results of the multi-training are going to be saved in a folder in the same output directory as the one for training.
The name of the multi-training folder is set automatically to : `MULTI_<date>_<var1-name>_<var2-name>_...`

#### config
The multi-training is a bit different to use than the pre-processing and the training. 
In order launch a multi-training, the setting up is done directly in the script `./multi_training.py` between the comments:
```
# ===========================================
# ====== SETTING UP THE MULTI-TRAINING ======
# ===========================================
"""
  <instructions>
"""
.
. <setting up herer>
.
# ===========================================
# ===========================================
```
#### run
Once everyting is set, the multi-training is done by running the file `./multi_training.py`.
WARNING! Do not change or run the script `.multi_training_src.py`!

### Model
The trained models and the script of the corresponding class, called `ASPP_Classifier`, are located in `./models/`. The model has the following characteristics:
- Can be trained in two modes:
 - `multi` which allows to do multi-class classification among the 6 following classes: _bare_, _terrace_, _spontaneous_, _extensive_, _lawn_ and _intensive_
 - `binary` which allows to do binary classification to detect if the roof is _bare_ or _vegetated_
- Customable backbone and ASPP modulus.

#### config
In order to modify the model's mode: the following two parameters need to be changed:
- `training.model.model_type`: choose the value between _"multi"_ and _"binary"_
- `preprocessing.inputs.class_labels_dir`: choose the path to the cooresponding csv file.
The model config can be set in the file `./config/training.yaml`. For more information on the different hyper-parameters, a `README.md` file is prensent a the root of the folder `./config/`.

### Inference
Once a model has been trained, the inference can be performed on new, unlabelized data.

#### inputs
The sources used to do inference are:
- rasters from measurements taken by plane in the NIR,R,G,B bands
- building's footprints as geometries
- A pre-trained model saved as a _.tar_ file

#### outputs
the results of the training are going to be saved in a folder named `<date>_inference_<suffixe>` (e.g. _20241109_inference_test_) in the specified location in the config file. For more information, look at the section [Results](#results).

#### config
The config file contains the paths to the different inputs and outputs for the inference, as well as some config parameters of the preprocessing and training to overwright.

#### run
Once the config file is set, the inference can then be done by running the file `inference.py`

### BigData process
The dataset used for inference and training can be too big for the working space. In order to avoid having to preprocess huge dataset in one go, the bigdata process allow to batch on those dataset for training and inference.

#### inputs
This process acts as an overlay on the chosen application (_inference_ or _training_). Therefore, the inputs are going to be the ones corresponding to the application.

#### outputs
The output of a process through this script is going to be placed in a folder named `<date>_BIGDATA_<mode>_<suffixe>` in the specified location in the config file.

#### config
The config file of bigdata only contains informations specific to the overlay (the mode, the batch_size, the suffixe, etc). Contrary to the inference, this script will require to config also the corresponding config files (`preprocessing.yaml`, `training.yaml` or `inference.yaml`)  

#### run
Once all config files are set, the bigdata process is done by running the file `bigdata_process.py`.

### Results
The results of trainings and multi-trainings are loaded, by default, in the folders `./results/trainings`, `./results/inferences` and `./results/bigdata` (can be changed in the corresponding .yaml files).

Regarding the training, the architecture presents as following:
- \<name_of_training_folder\>
  - images/ - where the performances and training graphs are saved
  - logs/ - where the datas used to create the graphs + sample logs (pred vs gt for each sample for each every epoch) are saved
  - models/ - where the models that performed best on each metric + the last model (for pretrained model use) are loaded
 
Regarding the multi-training, the architecture presents as following:
- \<name_of_multi-training_folder\>
  - best_results - the best results of each training for each metric
  - images - the graph of the multi-training performances. The type of graph will depend on the multi-training mode and on the number of variables
  - single_trainings - where all the single trainings are made. Their individual architecture is the one of a training

Regarding the inference, the architecture presents as following:
- \<name_of_inference_folder\>
  - <name_of_original_polygon_file>_preds.csv
  - <name_of_original_polygon_file>_preds.gpkg

Regarding bigdata, the architecture presents as following:
- \<name_of_big_data_folder\>
  - batch_0\
  - ...
  - batch_N\
    - <name_of_inference_on_batch_N> - if mode = inference
    - <name_of_trainging_on_batch_N> - if mode = training


