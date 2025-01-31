# %%
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

#os.environ['no_proxy'] = '127.0.0.1'
#matplotlib.use('module://backend_interagg')

# %% plot roc curves
'''from sklearn.metrics import roc_curve, auc

labels = np.load("labels.npy")
probas = np.load("probas.npy")
fpr, tpr, thres = roc_curve(labels, probas[:,1])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(nrows=1)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic curve')
ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
ax.legend(loc="best")
ax.grid(alpha=.4)
plt.savefig("roc_curve.png")
plt.show()'''


# %%
def plot_progress(progress_df, title, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].plot(progress_df.index, progress_df.train_loss, label="training loss")
    ax[0].plot(progress_df.index, progress_df.val_loss, label="validation loss")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(progress_df.index, progress_df.train_accuracy, label="training accuracy")
    ax[1].plot(progress_df.index, progress_df.val_accuracy, label="validation accuracy")
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path + "/train_val_curves.png")
    plt.show()


# %%
def plot_quant_vs_class(progress_df: list, names: list):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for df, name in zip(progress_df, names):
        ax[0, 0].plot(df.index, df.train_loss, label="training loss %s" % name)
        ax[0, 1].plot(df.index, df.train_accuracy, label="training accuracy %s" % name)
        ax[1, 0].plot(df.index, df.val_loss, label="validation loss %s" % name)
        ax[1, 1].plot(df.index, df.val_accuracy, label="validation accuracy %s" % name)

    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()
    ax[0, 1].set_xlabel('Epochs')
    ax[0, 1].set_ylabel('Accuracy')
    ax[0, 1].legend()
    ax[1, 0].set_xlabel('Epochs')
    ax[1, 0].set_ylabel('Loss')
    ax[1, 0].legend()
    ax[1, 1].set_xlabel('Epochs')
    ax[1, 1].set_ylabel('Accuracy')
    ax[1, 1].legend()

    plt.suptitle("Comparison of classical and quantum networks")
    plt.tight_layout()
    plt.show()


#%%
'''df1, df2, df3, df4, df5, df6 = pd.read_csv("/data/maureen/var-quantum-exp/higher_order_enc_strong_10_workers/progress.csv"), \
                pd.read_csv("/data/maureen/var-quantum-exp/higher_order_enc/progress.csv"), \
                pd.read_csv("/data/maureen/var-quantum-exp/tried_1_conv_batch_64/progress.csv"), \
                pd.read_csv("/data/maureen/var-quantum-exp/Sequence_RX_CNOTs_docker/progress.csv"), \
                pd.read_csv("/data/maureen/var-quantum-exp/tried_1_conv_batch_2/progress.csv"), \
                pd.read_csv("/data/maureen/var-quantum-exp/Sequence_RX_CNOTs_docker_batch2/progress.csv")

plot_quant_vs_class([df1, df2, df3, df4, df5, df6], ["Strong_Entangled", "Random_Layer", "Classical", "Sequence_RX_CNOTs", "Classical_batch2", "Sequence_RX_CNOTs_batch2"])

#%%
df1, df2 = pd.read_csv("/data/maureen/var-quantum-exp/tried_1_conv_batch_2/progress.csv"), \
                pd.read_csv("/data/maureen/var-quantum-exp/Sequence_RX_CNOTs_docker_batch2/progress.csv")

plot_quant_vs_class([df1, df2], ["Classical_batch2", "Sequence_RX_CNOTs_batch2"])'''

