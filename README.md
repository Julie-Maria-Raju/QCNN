# QCNN
Learnability of Noise and Error Mitigation in Hybrid Quantum-Classical CNN

This repository is the implementation of a Quantum CNN.

## Installation 

The following section describes the necessary steps to develop and execute your code with VSCode (an IDE specifically designed for Python) in a virtual machine within a Linux virtual environment.

## How to use?
`train.py` trains a hybrid model on the specified settings in `settings.yaml`.

#### Inputs:
- Use one of the yaml files in ./settings as a baseline and change parameters as desired.
- Run `python3 train.py --config-name <name of the yaml file in ./settings>`, e.g. `python3 train.py --config-name settings_2D_breastMNIST.yaml`.'python3 train.py --config-name settings_2D_medMNIST.yaml'
- Multiple runs can be done in one command line using the MULTITUN argument, e.g. `python3 train.py --config-name settings_2D_breastMNIST.yaml hydra.mode=MULTIRUN ++PARAMS.all_seeds=1,2,3,4,5` to run the same settings with five different seeds. See the hydra documentation if needed.

#### Outputs:
- SAVE_DIR/
    - SAVE_NAME/EXP_NAME/
        - circuit.txt  *#text file with a drawing of the quantum circuit*
        - events.out.tfevents  *#tensorboard logs*
        - train_val_curves.png  *#png image of the training and validation train_val_curves*
        - progress.csv  *#csv summary of training and validation progress*
        - checkpoints/  *#checkpoints files*
     
## File Description
 - calculations - custom_entanglements.py - Sequence_RX_CNOTs
 - data - medmnist.py
 - encoders - custom_higher_order_encoder.py
 - settings - settings_2D_medMNIST.yaml 

This Repo is still under development.
