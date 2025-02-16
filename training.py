import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date
import time
import torch
import torchvision.transforms as transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from preprocessing_multiproc import preprocess
from src.dataset import GreenRoofsDataset
from src.dataset_utils import ToTensor, Normalize
from models.ASPP_Classifier import ASPP_Classifier
from models.ASPP_Classifier_no_backbone import ASPP_no_backbone
from models.ASPP_Classifier_no_aspp import ASPP_no_aspp
from models.ASPP_Classifier_no_aspp_no_backbone import ASPP_no_aspp_no_backbone
from models.ASPP_Classifier_no_standardization import ASPP_Classifier_no_standardization
from models.ASPP_GlobalStats_Classifier import ASPP_GlobStat_Classifier
from src.visualization import show_confusion_matrix, show_log_train, show_precision_recall, show_gradient
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


def log_gradients_to_tensorboard(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(param.grad)
            print(param.grad.shape)
            # writer.add_histogram(f'Gradients/{name}', param.grad, step)


# def save_gradients_hook(module, grad_input, grad_output):
#     # Store or log the gradients for this module
#     if grad_output[0] is not None:
#         print(f"Gradients at {module}: {grad_output[0].norm().item()}")

gradient_data = defaultdict(list)  # Store gradients for each layer
def save_gradients_hook(module, grad_input, grad_output):
    # Store or log the gradients for this module
    if grad_output[0] is not None:  # Check if gradient is computed
        gradient_data[module].append(grad_output[0].detach().cpu())


def train_epoch(epoch, modes, dataloaders, optimizer, criterion, model, scheduler, dataset_sizes):
    """
    Train the model for one epoch.

    Args:
        - epoch (int): Current epoch number.
        - modes (list): List of modes (e.g., ['train', 'val']).
        - dataloaders (dict): Dictionary of dataloaders for each mode.
        - optimizer (torch.optim.Optimizer): Optimizer for the model.
        - criterion (callable): Loss function.
        - model (torch.nn.Module): The model to train.
        - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        - dataset_sizes (dict): Dictionary with sizes of datasets for each mode.

    Returns:
        - tuple: Contains epoch losses, accuracies, precision, recall, F1 scores,
               F2 scores, F0.5 scores, confusion matrix, and sample result logs.
    """
    confusion_matrix = None
    pred_tot = {phase: [] for phase in modes}
    target_tot = {phase: [] for phase in modes}
    epoch_losses = {}
    epoch_accs = {}
    epoch_precision = {}
    epoch_recall = {}
    epoch_f1 = {}
    epoch_f2 = {}
    epoch_f05 = {}

    samp_res_logs = {}

    # Each epoch has a training and validation phase
    for phase in modes:
        samp_res_logs[phase] = {'egid': [], 'target': [], 'pred': [], 'pred_conf': []}
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for _, data in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
            inputs, [targets, egids] = data['image'], data['label']
            [images, globalStats] = inputs
            images, globalStats, targets = images.cuda(), globalStats.cuda(), targets.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(images, globalStats)
                preds = outputs.data.max(1)[1]
                preds_conf = outputs.data.max(1)[0]
                loss = criterion(outputs.float(), targets)
                
                pred_tot[phase].append(preds.tolist())
                target_tot[phase].append(targets.tolist())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # statistics
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == targets.data)

            # samples analysis
            samp_res_logs[phase]['egid'].append(egids)
            samp_res_logs[phase]['target'].append(targets.tolist())
            samp_res_logs[phase]['pred'].append(preds.tolist())
            samp_res_logs[phase]['pred_conf'].append(preds_conf.tolist())

        # metrics
        pred_tot[phase] = [x for row in pred_tot[phase] for x in row]
        target_tot[phase] = [x for row in target_tot[phase] for x in row]

        epoch_losses[phase] = running_loss / dataset_sizes[phase]
        epoch_accs[phase] = accuracy_score(target_tot[phase], pred_tot[phase])
        epoch_precision[phase] = precision_score(y_true=target_tot[phase], y_pred=pred_tot[phase], average='macro', zero_division=0.0)
        epoch_recall[phase] = recall_score(target_tot[phase], pred_tot[phase], average='macro', zero_division=0.0)
        epoch_f1[phase] = f1_score(target_tot[phase], pred_tot[phase], average='macro', zero_division=0.0)
        epoch_f2[phase] = fbeta_score(target_tot[phase], pred_tot[phase], average='macro', zero_division=0.0, beta=2)
        epoch_f05[phase] = fbeta_score(target_tot[phase], pred_tot[phase], average='macro', zero_division=0.0, beta=0.5)
   
        print(f'{phase} Loss: {epoch_losses[phase]:.4f} Acc: {epoch_accs[phase]:.4f} F1: {epoch_f1[phase]:.4f} F2: {epoch_f2[phase]:.4f} F05: {epoch_f05[phase]:.4f} Precision: {epoch_precision[phase]:.4f} Recall: {epoch_recall[phase]:.4f}')

        # save confusion matrix if validation mode 
        if phase == 'val':           
            confusion_matrix = { 'pred': pred_tot, 'target': target_tot}

        # save samples_logs results
        samp_res_logs[phase]['egid'] = [x for row in samp_res_logs[phase]['egid'] for x in row]
        samp_res_logs[phase]['target'] = [x for row in samp_res_logs[phase]['target'] for x in row]
        samp_res_logs[phase]['pred'] = [x for row in samp_res_logs[phase]['pred'] for x in row]
        samp_res_logs[phase]['pred_conf'] = [x for row in samp_res_logs[phase]['pred_conf'] for x in row]

    return epoch_losses, epoch_accs, epoch_precision, epoch_recall, epoch_f1, epoch_f2, epoch_f05, confusion_matrix, samp_res_logs


def train_model(dataloaders, modes, dataset_sizes, model, criterion, optimizer, scheduler, batch_size, from_pretrained, num_epochs=25, dirs=""):
    """
    Train the model for a specified number of epochs.

    Args:
        - dataloaders (dict): Dictionary of dataloaders for training and validation.
        - modes (list): List of modes (e.g., ['train', 'val']).
        - dataset_sizes (dict): Dictionary with sizes of datasets for each mode.
        - model (torch.nn.Module): The model to train.
        - criterion (callable): Loss function.
        - optimizer (torch.optim.Optimizer): Optimizer for the model.
        - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        - batch_size (int): Batch size for training.
        - from_pretrained (bool): Whether to load a pre-trained model.
        - num_epochs (int, optional): Number of epochs to train. Default is 25.
        - dirs (str, optional): Directory paths for saving results.

    Returns:
        - tuple: Contains confusion matrices, metrics DataFrames, best metrics, and best epochs.
    """
    since = time.time()

    # create metrics contrainers
    metrics = {key: [] for key in modes}

    # initialize variables for training logs
    best_epochs = {'OA': 0, 'F1': 0,  'F2': 0.0,  'F05': 0.0, 'Precision': 0, 'Recall': 0}
    best_metrics = {'OA': 0.0, 'F1': 0.0,  'F2': 0.0,  'F05': 0.0, 'Precision': 0.0, 'Recall': 0.0}
    dict_conf_mats = {}
    dict_samp_logs = {}

    first_epoch = 0

    # sarting from existing model
    if from_pretrained:
        assert(os.path.exists("./models/pretrained/model.tar"))
        checkpoint = torch.load('./models/pretrained/model.tar', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        first_epoch = checkpoint['epoch'] + 1

    # coherency test
    if first_epoch >= num_epochs:
        raise ValueError("starting epoch is higher than ending epoch!!")
    
    # start training
    for epoch in range(first_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_losses, epoch_accs, epoch_precision, epoch_recall, epoch_f1, epoch_f2, epoch_f05, confmat, samp_logs_epoch = train_epoch(
            epoch, modes, dataloaders, optimizer, criterion, model, scheduler, dataset_sizes, 
            )
        current_metrics = {'OA': epoch_accs['val'], 'F1': epoch_f1['val'], 'F2': epoch_f2['val'], 'F05': epoch_f05['val'], 'Precision': epoch_precision['val'], 'Recall': epoch_recall['val']}

        # save models and metrics if best accuracy
        for metric in best_metrics.keys():
            if current_metrics[metric] > best_metrics[metric]:
                # save model
                print(f"Best {metric}. Saving model...")
                best_metrics[metric] = current_metrics[metric]
                best_epochs[metric] = epoch
                torch.save({
                    'epoch': epoch,
                    'batch_size': batch_size,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': epoch_accs['val'],
                    'val_loss': epoch_losses['val'],
                    'val_f1': epoch_f1['val'],
                    'train_accuracy': epoch_accs['train'],
                    'train_loss': epoch_losses['train'],
                    'train_f1': epoch_f1['train'],
                }, os.path.join(dirs['models_dir'], f"model_{metric}.tar"))

        if epoch == num_epochs-1:   # save last model
            print(f"Saving last epoch's model...")
            torch.save({
                    'epoch': epoch,
                    'batch_size': batch_size,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': epoch_accs['val'],
                    'val_loss': epoch_losses['val'],
                    'val_f1': epoch_f1['val'],
                    'train_accuracy': epoch_accs['train'],
                    'train_loss': epoch_losses['train'],
                    'train_f1': epoch_f1['train'],
                }, os.path.join(dirs['models_dir'], f"model_last.tar"))

        # add samp_logs of epoch to dict
        dict_samp_logs[epoch] = samp_logs_epoch

        # add confmat of epoch to dict
        df_confmat = pd.DataFrame(confmat)
        dict_conf_mats[epoch] = df_confmat

        # add metrics of epoch to dict
        for phase in modes:
            metrics[phase].append([epoch, epoch_losses[phase], epoch_accs[phase], epoch_f1[phase], epoch_f2[phase], epoch_f05[phase], epoch_precision[phase], epoch_recall[phase]])
        
        # save samples res
        with open(os.path.join(dirs['log_dir'], 'samp_logs.pickle'), 'wb') as output_file:
            pickle.dump(dict_samp_logs, output_file)
    
        # save confusion matrices
        with open(os.path.join(dirs['log_dir'], "confusion_matrices.pickle"), 'wb') as file:
            pickle.dump(dict_conf_mats, file)

    # save best results
    pd.DataFrame([[metric, best_metrics[metric], best_epochs[metric]] for metric in best_epochs.keys()], columns=['metric', 'best_val', 'epoch']).to_csv(
        os.path.join(dirs['log_dir'], "best_results.csv"), sep=';', index=False)

    # save metrics
    df_metrics = {}
    for phase in modes:
        df_metrics[phase] = pd.DataFrame(metrics[phase], columns=["epoch", "loss", "OA", "f1", "f2", "f05", "precision", "recall"])
        df_metrics[phase].to_csv(os.path.join(dirs['log_dir'], f"metrics_{phase}.csv"), index=False, sep=";")

    # print time to process
    time_elapsed = time.time() - since
    n_hours = int(time_elapsed / 3600)
    n_min = int((time_elapsed % 3600) / 60)
    n_sec = int(time_elapsed - n_hours * 3600 - n_min * 60)
    print(f'Training complete in {n_hours}:{n_min}:{n_sec}\n')

    # print best metrics
    for metric, val in best_metrics.items():
        print(f'Best {metric}: {round(val,2)}')

    return dict_conf_mats, df_metrics, best_metrics, best_epochs


def train(cfg:DictConfig):
    """
    Trains a deep learning model for semantic classification.

    Args:
        - cfg (DictConfig): Configuration containing training parameters and paths.

    Returns:
        - None
    """
    print("-----------------")
    # Test cuda compatibility and show torch versions
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        print("Cuda available")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("torch version : " + torch.__version__)
    print('device : ' + str(DEVICE))
    print("-----------------")
    
    torch.manual_seed(42)

    # ========================
    # ==== INITIALISATION ====
    # ========================

    # load arguments
    INPUTS = cfg['training']['inputs']
    DATASET_ROOT = INPUTS['dataset_root']

    OUTPUTS = cfg['training']['outputs']
    RES_DIR = OUTPUTS['res_dir']
    FOLDER_NAME_SUFFIX = OUTPUTS['folder_name_suffix']

    PROCESSES = cfg['training']['processes']
    DO_PREPROCESSING = PROCESSES['do_preprocessing']
    DO_GRADIENT_TRACKING = PROCESSES['do_gradient_tracking']

    PARAMETERS = cfg['training']['parameters']
    NORM_BOUNDARIES = np.array(PARAMETERS['norm_boundaries'])
    NUM_EPOCHS = PARAMETERS['num_epochs']
    BATCH_SIZE = PARAMETERS['batch_size']
    NUM_WORKERS = PARAMETERS['num_workers']
    DATA_FRAC = PARAMETERS['data_frac']
    TRAIN_FRAC = PARAMETERS['train_frac']
    LEARNING_RATE = PARAMETERS['learning_rate']
    WEIGHT_DECAY = PARAMETERS['weight_decay']
    FROM_PRETRAINED = PARAMETERS['from_pretrained']

    MODEL = cfg['training']['model']
    MODEL_TYPE = MODEL['model_type']
    BACKBONE = MODEL['backbone']
    BACKBONE_NUM_LEVELS = MODEL['backbone_num_levels']
    BACKBONE_NUM_LAYERS = MODEL['backbone_num_layers']
    ASPP_ATROUS_RATES = MODEL['aspp_atrous_rates']
    DROPOUT_FRAC = MODEL['dropout_frac']
    MODEL_SRC = MODEL['model_src']

    TRAINING_MODE = cfg['training']['training_mode']
    PREPROCESSING = cfg['preprocessing']
    DO_RANGELIMIT = PREPROCESSING['processes']['do_rangelimit']
    DO_GLOBAL_STATS = PREPROCESSING['processes']['do_global_stats']
    RANGELIMIT_MODE = PREPROCESSING['metadata']['rangelimit_mode']
    RANGELIMIT_THRESHOLD = PREPROCESSING['metadata']['rangelimit_threshold']
    # Do preprocessing if needed
    if DO_PREPROCESSING and TRAINING_MODE == 'single':
        print("Preprocessing:")
        preprocess(cfg['preprocessing'])

    # Create results architecture
    #   _create folder name
    if TRAINING_MODE == 'single':
        folder_name = date.today().strftime("%Y%m%d") + "_num_epoch_" + str(NUM_EPOCHS) + "_frac_" + str(int(DATA_FRAC * 100)) + "_" + FOLDER_NAME_SUFFIX
        new_folder_name = folder_name
        i = 1
        while os.path.exists(RES_DIR + new_folder_name):
            new_folder_name = folder_name + "_" + str(i)
            i += 1
        folder_name = new_folder_name + "/"
    elif TRAINING_MODE == 'multi':
        folder_name = FOLDER_NAME_SUFFIX
    else:
        raise ValueError("Invalid training.training_mode parameter")

    #   _create rest of architecture
    dirs = {}
    dirs['resroot_dir'] = os.path.join(RES_DIR, folder_name)
    dirs['log_dir'] = os.path.join(dirs['resroot_dir'], "logs/")
    dirs['models_dir'] = os.path.join(dirs['resroot_dir'], 'models/')
    dirs['images_dir'] = os.path.join(dirs['resroot_dir'], "images/")
    for dir in dirs.values():
        os.mkdir(dir)

    # save hyperparameters
    with open(os.path.join(dirs['log_dir'], "hyperparameters.yaml"), 'w+') as file:
        OmegaConf.save(config=cfg, f=file.name)

    # ========================
    # ==== TRAINING ==========
    # ========================

    # prepare transforms
    # _ set boundaries
    if RANGELIMIT_MODE == 'none':   # if no limiting, set rangelimit to the max possible
        RANGELIMIT_THRESHOLD = 2**16 - 1

    if DO_RANGELIMIT:   # if rangelimit, modifiy boundaries
        NORM_BOUNDARIES[0:4,1] = RANGELIMIT_THRESHOLD
        NORM_BOUNDARIES[5,1] = 3*RANGELIMIT_THRESHOLD

    # _ set transform
    transform = transforms.Compose([
        Normalize(NORM_BOUNDARIES),
        ToTensor(),
    ])

    # Create datasets for training & validation
    modes = ['train', 'val']
    datasets = {x: GreenRoofsDataset(DATASET_ROOT, 
                                    mode= x, 
                                    data_frac=DATA_FRAC,
                                    train_frac=TRAIN_FRAC,
                                    transform=transform,
                                    with_gs=DO_GLOBAL_STATS,
                                    ) for x in modes}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], 
                                                batch_size=BATCH_SIZE, 
                                                shuffle=(x == 'train'),
                                                num_workers=NUM_WORKERS,
                                                drop_last=True,
                                                ) for x in modes}

    dataset_sizes = {x: len(datasets[x]) for x in modes}
    class_names = datasets['train'].class_names

    # show infos about the datasets
    print(f"Datasets sizes:\n\
          - training set : {len(datasets['train'])} samples.\n\
          - validation set : {len(datasets['val'])} samples.")
    print("-" * 10)
    vals = {x: [] for x in modes}
    for mode in modes:
        for x in datasets[mode]:
            vals[mode].append(x['label'][0]) 
    print("Number of samples per class per dataset:")
    for i in list(set(vals['train'])):
        print('label ' + str(i))
        print(f"\t on training set: {vals['train'].count(i)}")
        print(f"\t on validation set: {vals['val'].count(i)}")
    print("-" * 10)

    # get model
    input_channels, img_size, _ = datasets['val'][0]['image'][0].size()
    #model = ASPP_Classifier(
    #model = ASPP_no_backbone(
    #model = ASPP_no_aspp(
    #model = ASPP_no_aspp_no_backbone(
    #model = ASPP_Classifier_no_standardization(
    model = ASPP_GlobStat_Classifier(
        input_channels=input_channels,
        output_channels=len(class_names),
        img_size=img_size,
        batch_size=BATCH_SIZE,
        backbone=BACKBONE,
        bb_levels=BACKBONE_NUM_LEVELS,
        bb_layers=BACKBONE_NUM_LAYERS,
        aspp_atrous_rates=ASPP_ATROUS_RATES,
        dropout_frac=DROPOUT_FRAC,
        mode=MODEL_TYPE,
    ).double().to(torch.device(DEVICE))
    num_param_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size : {num_param_model} parameters.")

    # initialize weights
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

    # Hooking gradients
    if DO_GRADIENT_TRACKING:
        # Register hooks on specific layers (e.g., Conv2d or Linear layers)
        model.backbone.register_full_backward_hook(save_gradients_hook)
        model.aspp.register_full_backward_hook(save_gradients_hook)
        model.mlp_conv.register_full_backward_hook(save_gradients_hook)
        model.mlp_gs.register_full_backward_hook(save_gradients_hook)
        model.mlp_merge.register_full_backward_hook(save_gradients_hook)

    # get class weights:
    targets = pd.read_csv(os.path.join(DATASET_ROOT,'dataset.csv'), delimiter=';')
    targets = targets['label'].to_numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets,
    )
    print('Weights : ', weights)
    print("-" * 10)
    class_weights = torch.tensor(weights, dtype=torch.float, device=torch.device(DEVICE))

    # Define criterion, optimizer and scheduler
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    model = model.to(torch.device(DEVICE))

    # training
    dict_conf_mats, df_metrics, best_metrics, best_epochs = train_model(dataloaders=dataloaders, 
                modes=modes, 
                dataset_sizes=dataset_sizes,
                model=model, 
                criterion=criterion, 
                optimizer=optimizer, 
                scheduler=scheduler,
                batch_size=BATCH_SIZE,
                from_pretrained=FROM_PRETRAINED,
                num_epochs=NUM_EPOCHS,
                dirs=dirs,
                )
    
    # ========================
    # ==== CREATE RESULTS ====
    # ========================
    
    # gradient tracking
    if DO_GRADIENT_TRACKING: 
        dict_gradients = {}
        modules = ['Backbone', 'ASPP', 'MLP-Conv', 'MLP-GS', 'MLP-Merging']
        for i, (module, grads) in enumerate(gradient_data.items()):
            # Flatten and concatenate all gradients collected
            all_grads = torch.cat([g.view(-1) for g in grads]).numpy()
            dict_gradients[modules[i]] = all_grads
        with open(os.path.join(dirs['log_dir'], 'gradient_logs.pickle'), 'wb') as gradfile:
            pickle.dump(dict_gradients, gradfile)
        show_gradient(dict_gradients, os.path.join(dirs['images_dir'],'gradients.png'), True, False)

        
    # training logs
    res_logs = {
        'epoch': df_metrics['train']['epoch'],
        'train_acc': df_metrics['train']['OA'],
        'train_loss': df_metrics['train']['loss'],
        'val_acc': df_metrics['val']['OA'],
        'val_loss': df_metrics['val']['loss'],
    }
    show_log_train(res_logs, os.path.join(dirs['images_dir'],'acc_loss_evolution.png'), True, False)

    # precision and recall
    show_precision_recall(df_metrics['val'], os.path.join(dirs['images_dir'], "precision_recall_logs.png"))

    # confusion matrices for each metric
    class_labels = pd.read_csv(os.path.join(DATASET_ROOT,'class_names.csv'), sep=';').drop_duplicates(subset='cat').cat.values
    for metric in best_metrics.keys():   
        pred = dict_conf_mats[best_epochs[metric]]['pred']['val']
        target = dict_conf_mats[best_epochs[metric]]['target']['val']
        show_confusion_matrix(target_src=os.path.join(dirs['images_dir'],f"confusion_matrix_epoch={best_epochs[metric]}_{metric}={round(best_metrics[metric], 2)}.png"),
                            y_pred=pred,
                            y_true=target, 
                            class_labels=class_labels,
                            title=f"Confusion Matrix - epoch={best_epochs[metric]} - {metric}={round(best_metrics[metric], 2)}",
                            do_save=True,
                            do_show=False
                            )


if __name__ == "__main__":
    # Retrieve parameters
    conf_preprocessing = OmegaConf.load('./config/preprocessing.yaml')
    conf_training = OmegaConf.load('./config/training.yaml')
    cfg = OmegaConf.merge(conf_preprocessing, conf_training)

    # train model
    train(cfg)
