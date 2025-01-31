import matplotlib.pyplot as plt
import pickle
import os

base_path = "/data/hanady.gebran/"
exp_path = "var-quantum-exp/exp1"

def plot_weights(all_weights,base_path,exp_path):
    "plots the evolution of the all_weights array for all the weights in a single plot and saves it to a file in the specified path"
    weights_shape = all_weights[0].shape
    for i in range(weights_shape[0]):
        for j in range(weights_shape[1]):
            weight_evolution = [all_weights[k][0][j].cpu()
                                for k in range(len(all_weights))]
            plt.plot(weight_evolution)
    plt.savefig(os.path.join(base_path, exp_path, 'weights_plot_all.pdf'))

def plot_weights_individual(all_weights,base_path,exp_path):
    "plots the evolution of the all_weights array for each weight individually and saves each plot to a separate file in the specified path"
    weights_shape = all_weights[0].shape
    for i in range(weights_shape[0]):
        for j in range(weights_shape[1]):
            weight_evolution = [all_weights[k][0][j].cpu()
                                for k in range(len(all_weights))]
            plt.plot(weight_evolution)
            plt.savefig(os.path.join(base_path, exp_path, f'weights_plots_{i}_{j}.pdf'))
            plt.clf()     

with open(os.path.join(base_path, exp_path, 'all_weights.pickle'), 'rb') as handle:
    all_weights = pickle.load(handle)
    plot_weights(all_weights,base_path,exp_path) #all weights together
    plot_weights_individual(all_weights,base_path,exp_path) #each weight alone