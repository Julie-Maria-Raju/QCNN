# %%
from ray_tune_module import train
import os
import time
from tabulate import tabulate
import yaml
from models.hybrid import *
from data.mnist import mnist_dataset
from data.medmnist import medmnist_dataset
from data.breastmnist import breast_mnist_dataset
import argparse
import sys
import hydra
import matplotlib.pyplot as plt
import pickle
import mitiq

MNIST_ONE_CONV = "mnist_one_conv"
MNIST_ONE_CONV_MULTI_OUT = "mnist_one_conv_multiOut"
MNIST_ONE_CONV_MULTI_OUT_MULTI_PASS = "mnist_one_conv_multiOut_multipass"
MNIST_TWO_CONV = "mnist_two_conv"
MNIST_TWO_CONV_MIDDLE = "mnist_two_conv_middle"

MEDICAL_2D = "medical_2D"
MEDICAL_3D = "medical_3D"
MEDICAL_3D_CONV_MIDDLE = "medical_3D_conv_middle"


def init(config):
    """
    Initialize a model and dataset by the given configurations.
    """

    quantum = config["QUANTUM"]
    model_dic = {"mnist": {
        MNIST_ONE_CONV: HybridModel_MNIST_one_conv,
        MNIST_TWO_CONV: HybridModel_MNIST_two_conv,
        MNIST_TWO_CONV_MIDDLE: HybridModel_MNIST_two_conv_middle,
        MNIST_ONE_CONV_MULTI_OUT: HybridModel_MNIST_one_conv_multiOut,
        MNIST_ONE_CONV_MULTI_OUT_MULTI_PASS: HybridModel_MNIST_one_conv_multiOut_multipass},  
        "medical": {
            MEDICAL_2D: HybridModel_medical_2D,
            MEDICAL_3D: HybridModel_medical_3D,
            #MEDICAL_3D_CONV_MIDDLE: HybridModel_medical_3D_conv_middle,
        },
    }
    
    if config["MODEL"] in model_dic["mnist"]:
        model = model_dic["mnist"][config["MODEL"]]
        if config["DATASET"] == "mnist":
            model = model(config, quantum=quantum)
            dataset = mnist_dataset(path="/qc-diag/mnist")
        elif config["DATASET"] == "med_mnist":
            #model_wo_noise = model(config, quantum=quantum, out_classes=11)
            model_with_noise = model(config, quantum=quantum, out_classes=11)
            dataset = medmnist_dataset(config["PARAMS"], train_size=1000, val_size=300) # Change here for datasie 1000,300
            
            config["PARAMS"]["magnitude"]=0.05
            model_with_noise = model(config, quantum=quantum, out_classes=11)
            
        elif config["DATASET"] == "breast_mnist":
            model = model(config, quantum=quantum, out_classes=2)
            dataset = breast_mnist_dataset(
                config["PARAMS"], download_dir=config.get("DATA_DIR", None))
    
        else:
            sys.exit("An MNIST model was chosen with a non-MNIST dataset. Exiting...")

    
    else:
        sys.exit("Please choose a valid model! Exiting...")

    if (
        config["MODEL"] == MEDICAL_3D_CONV_MIDDLE or config["MODEL"] == MEDICAL_3D
    ) and not config["PARAMS"]["do_3D_conv"]:
        sys.exit("A 3D model was chosen but do_3D_conv is set false. Exiting...")

    elif (
        not (config["MODEL"] == MEDICAL_3D_CONV_MIDDLE or config["MODEL"] == MEDICAL_3D)
        and config["PARAMS"]["do_3D_conv"]
    ):
        sys.exit("A 2D model was chosen but do_3D_conv is set true. Exiting...")

    return model_with_noise, dataset #model,model_wo_noise,


def plot_input_dist(dataset_to_plot, output_plot_path, xlabel):
    for i in range(len(dataset_to_plot)):
        try:
            images, labels, index = dataset_to_plot[i]
        except:
            images, labels = dataset_to_plot[i]
        if i==0:
            x_vector = images[0].ravel()
        else:
            x_vector = np.append(x_vector, images[0].cpu().detach().numpy().ravel())

    plt.figure()
    plt.hist(x_vector, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("occurrence")
    plt.savefig(output_plot_path)


def run_model(model, dataset, config):

    print("Neural Network Summary:")
    print(model)

    print("All parameters:")
    params = list(model.named_parameters())
    print(params)

    start_time = time.time()

    train(data=dataset, model=model, config=config)

    elapsed_time = time.time() - start_time
    print("time needed for training = ", elapsed_time)



@hydra.main(version_base=None, config_path="./settings", config_name="settings_2D_medMNIST.yaml") #settings_2D_breastMNIST.yaml
def main(config):
    if config["EXP_NAME"] == None:
        #config["EXP_NAME"] = "%s_%s_layers_%s_%s_seed_%s" % (
         #   config["PARAMS"]["calculation"], config["PARAMS"]["encoder"], config["PARAMS"]["circuit_layers"], config["PARAMS"]["rotation_factor"],config["PARAMS"]["all_seeds"])
        config["EXP_NAME"] = "%s_%s_layers_%s_%s_error_%s_seed_%s" % (
           config["PARAMS"]["calculation"], config["PARAMS"]["encoder"], config["PARAMS"]["circuit_layers"], config["PARAMS"]["rotation_factor"],config["PARAMS"]["magnitude"], config["PARAMS"]["all_seeds"])
    if(config["CHECKPOINT_DIR"] == str(None)):
        if os.path.exists(os.path.join(config["SAVE_DIR"], config["SAVE_NAME"], config["EXP_NAME"])):
            print('The output directory and experiment name already exist. Choose a different directory name! Exiting...')
            exit()

        else:
            os.makedirs(os.path.join(
                config["SAVE_DIR"], config["SAVE_NAME"], config["EXP_NAME"]))
    else:
        if (os.path.exists(os.path.join(config["SAVE_DIR"], config["SAVE_NAME"], config["EXP_NAME"]))==False):
            print('The output directory and experiment name does not already exist. Verify the directory name! Exiting...')
            exit()

    os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']

    model, dataset = init(config)

   
    run_model(model, dataset, config)
    

if __name__ == "__main__":
    main()
