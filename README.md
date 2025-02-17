# Deep learning base method for green roofs classification

## Installation
### downolading data
The different datasets (scratch tiles, tlm tiles, class_labels files and groundtruth polygons) can be downloaded following [this link](https://sftpgo.stdl.ch/web/client/pubshares/UZCZkgFu8V4dfFtyfrWpsc/browse) with the following credentials:
- login: cmarmy_user
- pwd: vegRoofs#24

The location `./data/sources/` has been set to receive them.
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
If incompatibilities happen, a list of commmand lines to install the different libraries is also present in the file 

`./libraries_installation_line.txt`.

### (docker)
not yet implemented..

## architecture
The architecture of the project is as following:
- ./
  - config/
    - config.yaml
    - preprocessing.yaml
    - training.yaml
  - data/
  - pretrained/
    - model.tar
  - results/
    - trainings/
    - trainings_archive/
  - src/
    - models/
    - dataset.py
    - dataset_utils.py
    - preprocess_utils.py
    - visualization.py
  - inference.py
  - multi_training.py
  - training.py
  - preprocessing.py
  - requirements.txt
  - libraries_installation_line.txt
 
Remark: If multi-trainings are done, the library _hydra_ will create a folder **multirun** at the root of the project. This folder can be ignored (and already is in the .gitignore).

## Documentation
### Preprocessing
The preprocessing consists in one script `./preprocess.py` driven by one configuration file `./config/preprocess.yaml`.

#### inputs
The sources used to create the dataset are:
- rasters from measurements taken by plane in the R,G,B,NIR bands
- building's footprints as geometries

#### outputs
The results from the preprocess should be a ready-to-use dataset. In order to do so, the main aim of the `preprocess.py` script is to clip the bbox of each polygons to the rasters in order to get a single sample per roof.

[//]: # (#### processes)
[//]: # (Once the sample is isolated from the intitial rasters, multiple process can be done to it in order to prepare it for the model:)
[//]: # (- Cropping: The sample is cropped on the geometry of the building's footprint.)
[//]: # (- NDVI layer: The ndvi is computed and added as a fifth layer.)
[//]: # (- luminosity layer: The luminosity layer is computed and added as a sixth layer)
[//]: # (- Masking based on NDVI: a mask is created based on a threshold applied to the NDVI layer.)
[//]: # (- Smoothing option: A k-nn based algorithm can be used to make the mask broader and smoother \(less isolated pixels\).)
[//]: # (- Size normalization: The sample's size are homogenized to a predefined standard size. \(e.g. 256x256, 512x512, 1024x1024, ...\))

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

[//]: # (#### process)
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

#### run
- Once the config file is set, the script `./training.py` can be run.

### Multi-training
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
### Model
The models is/are located in `./models/`. For the moment, there is only the model `ASPP_Classifier` whith the following characteristics:
- Can be trained in two modes:
 - `multi` which allows to do multi-class classification among the 6 following classes: _bare_, _terrace_, _spontaneous_, _extensive_, _lawn_ and _intensive_
 - `binary` which allows to do binary classification to detect if the roof is _bare_ or _vegetated_
- Customable backbone modulus with modifiable number of levels and layers per level.

#### architecture
Both of these models are based on the same architecture: _a backbone_, _the ASPP_  and _an MLP_. The backbone is a serie of convolutional layers that is costumable both in levels (the number of times the signal's cardinality is divided by two) and in layers (the number of convolutional layers per level).

#### config
In order to modify the model mode: the following two parameters need to be changed:
- `training.model.model_type`: choose the value between _"multi"_ and _"binary"_
- `preprocessing.inputs.class_labels_dir`: choose the path to the cooresponding csv file.
The model config can be set in the file `./config/training.yaml`. For more information on the different hyper-parameters, a `README.md` file is prensent a the root of the folder `./config/`.

Moreover, the model can set for multi-class or binary-class classification. In order to do so, two 

### Results
The results of trainings and multi-trainings are loaded, by default, in the folder `./results/trainings/` (can be changed in the `training.yaml` file).

The name of the multi-training folder is set automatically to : `MULTI_<date>_<var1-name>_<var2-name>_...`

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

### References
.. none yet
