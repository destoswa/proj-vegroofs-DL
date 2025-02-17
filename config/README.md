This folder contains the different config files to drive the preprocessing, trainings, fine-tuning and inference.

# preprocessing.yaml
this file contains the following architecture of parameters concerning the preprocessing:
- **preprocessing**:
  - **working_directory**: \<string\> - the absolute path to working directory
  - **inputs**:
    - **polygon_src**: \<string\> - the realtive path to the ground truth file containing the labeled polygons.
    - **rasters_dir**: \<string\> - the relative path to the root of the folder containing the .tiff raster files.
    - **class_labels_dir**: \<string\> - the relative path to the class labels information file (.csv).
  - **outputs**:
    - **output_dir**: \<string\> - the relative path to the directory created by the preprocessing.
  - **processes**:
    - **image_norm_style**: \<string\> - style of image normalization technique (between `stretching` and `resizing`)
    - **do_da_rotation**: \<bool\> - add data-augmentation by rotating the under-represented classes up to 3 times.
    - **do_da_flipping**: \<bool\> - add data-augmentation by flipping the under-represented classes up to 2 times.
    - **do_drop_overlapping**: \<bool\> - add the dropping of the samples that overlapp two tiff files in the raster_dir.
    - **do_drop_based_on_ndvi**: \<bool\> - add the dropping of bare samples with mean ndvi greater than 0.05 in order to remove samples with over-hanging trees.
    - **do_produce_tif**: <\bool\> - choose to also save a .tif version of the samples or not (only for visual purpose)
  - **metadata**:
    - **preprocess_mode**: \<string\> - to choose if the preprocessing is to do _training_ or _inference_
    - **batch_size**: \<int\> - if wanting to process the dataset by chunks (in order to ease the RAM usage), can set the size of each here. otherwise, set it to 0
    - **sample_size**: \<int\> - size in pixel of the samples
    - **epsg**: \<int\> - espg number for polygon projection
  - **security**:
    - **do_abort**: \<bool\> - add to abort every run of the preprocessing

# training.yaml
this file contains the following architecture of parameters concerning the training:
- **training**:
  - **working_directory**: \<string\> - the absolute path to working directory
  - **inputs**:
    - **dataset_root**: \<string\> - the relative path from the working_directory to the dataset
    - **pretrained_src**: <\string\> - the relative path to the pretrained model (as a .tar file)
  - **outputs**:
    - **res_dir**: \<string\> - the relative path from the working directory to the result directory
    - **folder_name_suffix**: \<string\> - the suffixe added to the training directory's name
  - **processes**:
    - **do_preprocessing**: \<bool\> - add the preprocessing before starting the training
  - **parameters**:
    - **norm_boundaries**: \<list of tuples\> - the boudaries of each band used for normalization
    - **data_frac**: \<float\> - the fraction of the dataset to use
    - **train_frac**: \<float\> - the fraction of the dataset to train on. the rest is used for validation
    - **num_epochs**: \<int\> - the number of epochs on which the model is trained
    - **batch_size**: \<int\> - the number of sample per batch
    - **num_workers**: \<int\> - the number of workers used to retrieve the data
    - **learning_rate**: \<float\> - the learning rate used by the optimizer
    - **weight_decay**: \<float\> - the weight decay used by the optimizer
    - **from_pretrained**: <\bool\> - train from pretrained model if true. From scratch otherwhise
  - **training_mode**: \<string\> - parameter used by the multi-training - do not change
  - **model**:
    - **model_type**: \<string\> - the type of model for either multi-class or binary-class classification
    - **backbone_num_levels** \<int\> - the number of levels of the backbone
    - **backbone_num_layers**: \<int\> - the number of layers per levels of the backbone
    - **aspp_atrous_rates**: \<string\> - the rates of dilation of the different convolution layers of the aspp module
    - **dropout_frac**: \<float\> - the fraction of dropped weights at different positions in the model.

# config.yaml
  not to be modified by users.

# inference.yaml
this file contains the following architecture of parameters concerning the infering:
- **inference**:
  - **working_directory**: \<string\> - the absolute path to working directory
  - **inputs**:
    - **polygon_src**: \<string\> - the realtive path to the ground truth file containing the labeled polygons.
    - **rasters_dir**: \<string\> - the relative path to the root of the folder containing the .tiff raster files.
    - **class_labels_dir**: \<string\> - the relative path to the class labels information file (.csv).
    - **model_src**: \<string\> - the relative path to the model's save
  - **outputs**:
    - **dataset_dir**: \<string\> - the relative path to the directory created by the preprocessing.
    - **preds_dir**: \<string\> - the relative path to the folder that is going to receive the results
    - **folder_name_suffix**: \<string\> - the suffixe to the folder that is going to be created and stored in the results
  - **processes**:
    - **do_preprocessing**: \<bool\> - add the preprocessing before starting the inference
  - **parameters**:
    - **data_frac**: \<float\> - the fraction of the dataset to use
    - **mode**: \<string\> - the mode to use between _training_ and _inference_
  - **preprocess_overright**:
    - _parameters of preprocess to overright_
  - **training_overright**:
    - _parameters of preprocess to overright_

# bigdata.yaml
this file contains the following architecture of parameters concerning the bigdata processing:
- **bigdata**:
  - **working_directory**: \<string\> - the absolute path to working directory
  - **mode**: \<string\> - the chosen mode between `trainings` and `inference`
  - **batch_size**: \<int\> - the number of samples to process per batch
  - **outputs**:
    - **results_dir**: \<string\> - the realtive path to the folder where the results of bigdata processes are saved
    - **results_suffixe**: \<string\> - the suffixe added to the directory

