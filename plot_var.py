#%%
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
os.environ['no_proxy'] = '127.0.0.1'
#matplotlib.use('module://backend_interagg')

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

plt.setp(ax1, xticks=np.arange(0, 20, 1))
plt.setp(ax2, xticks=np.arange(0, 20, 1))
plt.setp(ax3, xticks=np.arange(0, 20, 1))
plt.setp(ax4, xticks=np.arange(0, 20, 1))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 14})
#ax1.set_ylim(ymin=0.3, ymax=0.6)
#ax2.set_ylim(ymin=0.3, ymax=0.6)

root = "/data/julie.maria.raju/qc-diag/results/"

"""exps = ["OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ ",
        "amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.0005_seed_",
        #"amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.05_seed_",
        #"amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.07_seed_",
        #"amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.1_seed_",
        #"amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.2_seed_",
        #"amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.3_seed_",
        #"amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.4_seed_",
        #"amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.5_seed_"
        ]"""
exps = ["OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ ",
    "amplitudeDamping_0.5_trained_with_noise/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.5_seed_",
     "amplitudeDamping__OrganaMNIST_300/Sequence_RX_CNOTs_Custom_Higher_Order_Encoder_layers_1_0.785_error_0.5_seed_"   ]
exps_labels = ["Without_noise","trained and validated with noise" ,"trained_wo_noise and validated with noise"] 
dir_names = [root + name for name in exps]

end_df = pd.DataFrame()
end_df["epochs"] = range(10)   #change range for epoch
for i in range(len(dir_names)):
    dir_name = dir_names[i]
    exp = exps[i]
    exp_label = exps_labels[i]
    dfs= []

    for exp_name in range(1,3):
        #exp_name += 1
        
            csv_file = os.path.join(dir_name + str(exp_name), "progress.csv")  # change csv file name
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df = df.head(10)
                val_loss = "val_loss"
                val_acc = "val_accuracy"
                dfs.append(df)
            else:
                print(f" The path doesn't exists...{csv_file}")
            
            val_csv_file = os.path.join(dir_name + str(exp_name), "validation_results.csv")  # change csv file name
            if os.path.exists(val_csv_file):
                df = pd.read_csv(val_csv_file)
                val_loss = "ptl/val_loss"
                val_acc = "ptl/val_accuracy"
                dfs.append(df)
            else:
                print(f" The path doesn't exists...{val_csv_file}")
            
        


    metrics = [val_loss, val_acc]
    #metrics = ["ptl/val_loss", "ptl/val_accuracy"]
        # ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]

    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = {}
        for df in dfs:
            for epoch in range(len(df)):
                if epoch not in metrics_dict[metric]:
                    metrics_dict[metric][epoch] = []
                metrics_dict[metric][epoch].append(df[metric].iloc[epoch])

    mean_metrics = {}
    std_metrics = {}
    for metric in metrics:
        mean_metrics[metric] = []
        std_metrics[metric] = []
        for epoch in range(len(df)):
            #print('metric {}: {}'.format(metric, metrics_dict[val_loss][epoch]))
            mean_metrics[metric].append(np.mean(metrics_dict[metric][epoch]))
            std_metrics[metric].append(np.std(metrics_dict[metric][epoch]))
    #print(mean_metrics)

    epochs = [epoch for epoch in range(len(df))]

    """ y = mean_metrics["train_loss"]
    ci = std_metrics["train_loss"]
    ax1.plot(epochs, y, label=exp_label)
    ax1.fill_between(epochs, np.subtract(y, ci), np.add(y, ci), alpha=.1)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right', fontsize=9)"""

    y = mean_metrics[val_loss]
    ci = std_metrics[val_loss]
    #print(ci)
    ax2.plot(epochs, y, label=exp_label)
    ax2.fill_between(epochs, np.subtract(y, ci), np.add(y, ci), alpha=.2)
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Val_Loss')
    #ax2.legend( fontsize=9)



    """y = mean_metrics["train_accuracy"]
    ci = std_metrics["train_accuracy"]
    ax3.plot(epochs, y, label=exp_label)
    ax3.fill_between(epochs, np.subtract(y, ci), np.add(y, ci), alpha=.1)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend(loc='lower right', fontsize=9)"""

    y = mean_metrics[val_acc]
    ci = std_metrics[val_acc]
    ax4.plot(epochs, y, label=exp_label)
    ax4.fill_between(epochs, np.subtract(y, ci), np.add(y, ci), alpha=.2)
    ax4.set_ylim(0, 0.8)
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Val_Accuracy')
    #ax4.legend(fontsize = 9)

    

#fig1.suptitle("Training loss")
"""fig1.tight_layout()
fig1.savefig(root + "var_curves_train_loss.pdf", bbox_inches='tight')
fig1.show()"""

#fig1.suptitle("Validation loss")
#ax2.plot(epochs, noise_val_loss_amplitude, label= 'Amplitude damping 0.05')
#ax2.plot(epochs, noise_val_loss_dephasing, label= 'Dephasing 0.05')
#ax2.plot(epochs, noise_val_loss_readout, label= 'Readout 0.05')
#ax2.plot(epochs, noise_val_loss_depolarising, label= 'Depolarising 0.05')
ax2.set_title('Validation loss over epochs for error rate=0.5')
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(root + "amplitude_3_curves_loss.png", bbox_inches='tight')
fig2.show()

#fig1.suptitle("Training accuracy")
"""fig3.tight_layout()
fig3.savefig(root + "var_curves_train_acc.pdf", bbox_inches='tight')
fig3.show()"""

#fig1.suptitle("Validation accuracy")
#ax4.plot(epochs, noise_val_accuracy_amplitude, label= 'Amplitude damping 0.05')
#ax4.plot(epochs, noise_val_accuracy_dephasing, label= 'Dephasing 0.05')
#ax4.plot(epochs, noise_val_accuracy_readout, label= 'Readout 0.05')
#ax4.plot(epochs, noise_val_accuracy_depolarising, label= 'Depolarising 0.05')
ax4.set_title('Validationaccuracy over epochs for error rate=0.5')
ax4.legend(fontsize=9)
fig4.tight_layout()
fig4.savefig(root +  "amplitude_3_curves.png", bbox_inches='tight')
fig4.show()