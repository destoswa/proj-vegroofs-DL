This folder contains the different config files to drive the preprocessing, trainings, fine-tuning and inference.

# preprocessing.yaml
this file contains the following architecture of parameters concerning the preprocessing:
- **preprocessing**:
  - **working_directory**: \<string\> - the path to the root of the project (useless until now)
  - **inputs**:
    - **polygon_src**: \<string\> - the path to the ground truth file containing the labeled polygons. Can be absolute or relative from the working_directory
    - **rasters_dir**: \<string\> - the path to the root of the folder containing the .tiff raster files. Can be absolute or relative from the working_directory
    - **class_labels_dir**: \<string\> - the path from to the class labels information file (.csv). Can be absolute or relative from the working_directory
  - **outputs**:
    - **output_dir**: \<string\> - the relative path from the working_directory to the directory created by the preprocessing
  - **processes**:
    - **do_rangelimit**: \<bool\> - only when using 16bits datasets - add range-limiting of the samples
    - **do_mask**: \<bool\> - only if also adding ndvi layer - add masking of the samples based on their ndvi value
    - **do_smooth_mask**: \<bool\> - only if also adding ndvi layer and masking - add smoothing of the mask results
    - **do_rotation_da**: \<bool\> - add data-augmentation by rotating the under-represented classes up to 3 times
    - **do_drop_overlapping**: \<bool\> - add the dropping of the samples that overlapp two tiff files in the raster_dir
    - **do_drop_based_on_ndvi**: \<bool\> - add the dropping of bare samples with mean ndvi greater than 0.05 in order to remove samples with over-hanging trees
  - **metadata**:
    - **sample_size**: \<int\> - size in pixel of the samples
    - **rangelimit_function**: \<string\> - only when using 16bits datasets - name of the range-limiting function to use 
    - **rangelimit_threshold**: \<int\> - only when using 16bits datasets - threshold to use for cropping the values in the range-limiting
    - **epsg**: \<int\> - espg number for polygon projection
  - **security**:
    - **do_abort**: \<bool\> - add to abort every run of the preprocessing

# training.yaml
this file contains the following architecture of parameters concerning the training:
- **training**:
  - **inputs**:
   - **dataset_root**: \<string\> - the relative path from the working_directory to the dataset
  - **outputs**:
    - **res_dir**: \<string\> - the relative path from the working directory to the result directory
    - **folder_name_suffix**: \<string\> - the name of the training directory name
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
    - **from_pretrained**: <\bool\> - train from `./pretrained/model.tar` if true. From scratch otherwhise
  
  - **training_mode**: \<string\> - parameter used by the multi-training - do not change

  - **model**:
    - **model_type**: \<string\> - the type of model for either multi-class or binary-class classification
    - **backbone**: \<string> - the type of backbone to use
    - **backbone_num_levels** \<int\> - the number of levels of the backbone
    - **backbone_num_layers**: \<int\> - the number of layers per levels of the backbone
    - **model_src**: \<string\> - the relative path from the working directory to the model

# config.yaml
  not to be modified by users.
