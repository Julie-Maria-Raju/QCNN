import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from lightning_modules.lightning_module_MNIST import ModelWrapper as ModelWrapper_MNIST
#from lightning_modules.lightning_module_LIDC_IDRI import ModelWrapper as ModelWrapper_LIDC_IDRI
from lightning_modules.lightning_module_MedMNIST import ModelWrapper as ModelWrapper_MedMNIST
#from lightning_modules.lightning_module_RSNA_ICH import ModelWrapper as ModelWrapper_RSNA_ICH
import os
import numpy as np
import pandas as pd
from plot import plot_progress
import json
import data as datasets
import pickle
import yaml
#from omegaconf import OmegaConf
import torch
import csv
### Ray tune module for hyperparameters optimization ###


def train(data, model, config, checkpoint_dir=None, num_epochs=None, num_gpus=1):
    if num_epochs is None:
        num_epochs = config["PARAMS"]["epochs_num"]
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # limit_train_batches=10,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=os.path.join(config["SAVE_DIR"], config["SAVE_NAME"]), name="", version=config["EXP_NAME"]),
    )

    if isinstance(data, datasets.mnist.MNISTDataset):
        ModelWrapper = ModelWrapper_MNIST
    elif isinstance(data, datasets.medmnist.MedMNISTDataModule):
        ModelWrapper = ModelWrapper_MedMNIST
    elif isinstance(data, datasets.breastmnist.BreastMNISTDataModule):
        ModelWrapper = ModelWrapper_MedMNIST
    else:
        print("TYPE IS", type(data))

    if(config["CHECKPOINT_DIR"] == str(None)):
        lightning_model = ModelWrapper(data, model, config)
    else:
        trainer = pl.Trainer(
        gpus=1, 
        max_epochs=num_epochs,
        logger=TensorBoardLogger(
        save_dir=os.path.join(config["SAVE_DIR"], config["SAVE_NAME"]),
        name="",
        version=config["EXP_NAME"]
        ),
        resume_from_checkpoint=config["CHECKPOINT_DIR"])
        ckpt = pl_load(
            os.path.join(config["CHECKPOINT_DIR"]),
            map_location=lambda storage, loc: storage)
        metrics_saved = pd.read_csv(os.path.join(config["SAVE_DIR"], config["SAVE_NAME"],config["EXP_NAME"],"progress.csv"))
        lightning_model = ModelWrapper(data, model, config, analysis={"train_loss": metrics_saved['train_loss'].tolist(), "train_accuracy": metrics_saved['train_accuracy'].tolist(),
                            "val_loss": metrics_saved['val_loss'].tolist(), "val_accuracy": metrics_saved['val_accuracy'].tolist()} )

    weights = torch.load(f"/data/julie.maria.raju/qc-diag/results/OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ 2/weights_epoch_5.pt")
    lightning_model.model.load_state_dict(weights)

    # For the medical dataset the data is handled by a custom LightningDataModule in data/LIDC_IDRI.py.
    # Therefore the fit() function needs as additional argument the dataset
    if isinstance(data, datasets.LIDC_IDRI.LIDC_IDRIDataModule):
        trainer.fit(lightning_model, data)
    # for MNIST no second argument is required in fit() since all data handling is done in lightning_module_MNIST.py
    # without a LightningDataModule
    else:
        trainer.fit(lightning_model)
    

    # Perform validation and capture the returned values
    # Define the file path for saving the CSV file
    """save_dir = os.path.join(config["SAVE_DIR"], config["SAVE_NAME"])
    save_path = os.path.join(save_dir, config["EXP_NAME"])
    csv_file_path = save_path + "/validation_results.csv"
    for i in range(num_epochs):
        weights = torch.load(f"/data/julie.maria.raju/qc-diag/results/amplitudeDamping_0.5_trained_with_noise/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.5_seed_2/weights_epoch_4.pt")
        #/data/julie.maria.raju/qc-diag/results/modc_organa_300/wo_noise/ConvPoolingLayerHad_Custom_Higher_Order_Encoder_layers_1_0.785_seed_3/weights_epoch_{i}.pt"
        lightning_model.model.load_state_dict(weights)
        validation_results = trainer.validate(model=lightning_model, dataloaders=lightning_model.data.val_dataloader())
        #validation_results = trainer.predict(model=lightning_model, dataloaders=lightning_model.data.val_dataloader())
        print(validation_results)
        
        for result in validation_results:
            result['epoch'] = i
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=validation_results[0].keys())
            if i == 0:
                writer.writeheader()
            writer.writerows(validation_results)"""

        #trainer.test(lightning_model, test_dataloaders=lightning_model.test_dataloader())
        #stacked_preds = lightning_model.stacked_preds.cpu().numpy()
        #stacked_y = lightning_model.stacked_y.cpu().numpy()
        #conf_mat = lightning_model.print_conf_matrix(stacked_preds, stacked_y)
        #print(conf_mat)
        #print((conf_mat[0, 0] + conf_mat[1, 1]) / (conf_mat[0, 0] + conf_mat[1, 1] + conf_mat[0, 1] + conf_mat[1, 0]))
    model.logits = lightning_model.logits
    # save plot
    # remove first data because validation end is called at the very beginning...
    lightning_model.analysis["val_loss"] = lightning_model.analysis["val_loss"][1:]
    lightning_model.analysis["val_accuracy"] = lightning_model.analysis["val_accuracy"][1:]

    # print("ANALYSIS", lightning_model.analysis)
    analysis_float = {}
    for metric, values in lightning_model.analysis.items():
        analysis_float[metric] = []
        for value in values:
            if isinstance(value, torch.Tensor):
                analysis_float[metric].append(value.item())
            else:
                analysis_float[metric].append(value)

    df = pd.DataFrame(analysis_float)
    quant_or_class = "Quantum" if config["QUANTUM"] == True else "Classical"
    title = "Training curves for experiment on " + type(data).__name__ + " with " + quant_or_class + " " + \
        type(model).__name__
    save_dir = os.path.join(config["SAVE_DIR"], config["SAVE_NAME"])
    save_path = os.path.join(save_dir, config["EXP_NAME"])
    plot_progress(progress_df=df, title=title, save_path=save_path)
    df.to_csv(save_path + "/progress.csv")
    # save weights
    if config["QUANTUM"] == True:
        with open(save_path + '/all_weights.pickle', 'wb') as handle:
            pickle.dump(lightning_model.all_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save the settings file
    config_yaml = OmegaConf.to_container(config, resolve=True)
    with open(os.path.join(save_path, 'settings.yaml'), 'w') as outfile:
        yaml.dump(config_yaml, outfile, default_flow_style=False)
    return trainer

# Not used
'''def train_tune(config, data, model, params, checkpoint_dir=None, num_epochs=None, num_gpus=1):

    if num_epochs is None:
        num_epochs = params["PARAMS"]["num_epochs"]

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=os.path.join(config["SAVE_DIR"], config["SAVE_NAME"]), name="", version=config["EXP_NAME"]),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "train_loss": "ptl/train_loss",
                    "train_accuracy": "ptl/train_accuracy",
                    "val_loss": "ptl/val_loss",
                    "val_accuracy": "ptl/val_accuracy"
                },
                filename="checkpoint",
                on="validation_end")
        ])

    if isinstance(data, datasets.mnist.MNISTDataset):
        ModelWrapper = ModelWrapper_MNIST
    elif isinstance(data, datasets.LIDC_IDRI.LIDC_IDRIDataModule):
        ModelWrapper = ModelWrapper_LIDC_IDRI
    elif isinstance(data, datasets.medmnist.MedMNISTDataModule):
        ModelWrapper = ModelWrapper_MedMNIST

    if checkpoint_dir:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        lightning_model = ModelWrapper._load_model_state(
            ckpt, config=config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        lightning_model = ModelWrapper(config=config, data=data, model=model)

    trainer.fit(lightning_model)


def tune_asha(config, data, model, params, num_samples=1, num_epochs=None, gpus_per_trial=1):
    if num_epochs is None:
        num_epochs = params["PARAMS"]["num_epochs"]

    config = config

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=params["PARAMS"]["num_epochs"],
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["train_loss", "test_accuracy", "val_loss", "val_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            data=data,
            model=model,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=os.path.join(params["SAVE_NAME"], params["EXP_NAME"]),
        local_dir=params["SAVE_DIR"],
        keep_checkpoints_num=1,
        checkpoint_score_attr="val_loss",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


def get_res(tuning_res, params, dataset):
    for exp_name in tuning_res:
        df = pd.DataFrame(tuning_res[exp_name])
        quant_or_class = "Quantum" if params["QUANTUM"] == True else "Classical"
        title = "Training curves for experiment on " + \
            type(dataset).__name__ + \
            " with " + quant_or_class + " model"
        # save_path = os.path.split(exp_name)[-1]
        plot_progress(progress_df=df, title=title, save_path=exp_name)
        with open(exp_name + '/all_settings.json', 'w') as fp:
            json.dump(params["PARAMS"], fp)'''
