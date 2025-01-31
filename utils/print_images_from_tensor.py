#%%
import matplotlib
matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['no_proxy'] = '127.0.0.1'

root = "/data/maureen/var-quantum-exp/breast_mnist_quant_4_images_saved_tensor"
num_epochs = 20
num_batches = 10
layers = ["after_first_conv"]

selected_images = [(9, 3), (8, 6), (7, 6), (1, 0), (3, 0), (2, 4)]
#%%
# first go through before first conv
selected_images = [(8, 6), (3, 0)]

for batch in range(num_batches):
    x = torch.load(os.path.join(root, 'tensors', 'tensors_epoch_%s_batch_%s_layer_%s.pt'%(0, batch, "before_first_conv")))
    label = torch.load(os.path.join(root, 'tensors', 'labels_epoch_%s_batch_%s_layer_%s.pt'%(0, batch, "before_first_conv")))
    print(x)
    print(x.shape[0])

    for j in range(x.shape[0]):
        img = x[j, :, :]
        # print("Shape before layer", img.shape)
        channels = img.shape[0]
        h, w = img.shape[1], img.shape[2]
        plt.figure()
        f, axarr = plt.subplots(channels, 1)

        # chose the scale
        get_min = -0.6
        get_max = 0.6
        # get_min = np.min(np.array(img.cpu().detach())) - 0.01
        # get_max = np.max(np.array(img.cpu().detach())) + 0.01
        if get_min == get_max:
            get_max = get_max + 0.1
        # print("min", get_min)
        # print("max", get_max)

        for channel in range(channels):
            img_chan = img[channel, :, :]
            # print("img_chan", img_chan.shape)

            try:
                im = axarr[channel].imshow(img_chan.cpu().detach(), cmap='gray', vmin=get_min, vmax=get_max)
                axarr[0].set_title("Batch %s index %s. Sample: %s" % (batch, j, "Malignant" if label[j] == 0 else "Benign"))
            except:
                im = axarr.imshow(img_chan.cpu().detach(), cmap='gray', vmin=get_min, vmax=get_max)
                axarr.set_title("Batch %s index %s. Sample: %s" % (batch, j, "Malignant" if label[j] == 0 else "Benign"))

            f.subplots_adjust(right=0.8)
            cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
            f.colorbar(im, cax=cbar_ax)
            if (batch, j) in selected_images:
                #plt.savefig(os.path.join(root, "images", 'original_image_batch_%s_index_%s.png' % (batch, j)))
                plt.show()
        plt.close()

#%% then print only selected images
selected_images = [(8, 6), (3, 0)]

if not os.path.exists(os.path.join(root, "images")):
    os.makedirs(os.path.join(root, "images"))

for selected_image in selected_images:
    batch = selected_image[0]
    j = selected_image[1]
    for layer in layers:
        xs = []
        labels = []
        for epoch in range(num_epochs):
            x = torch.load(os.path.join(root, 'tensors', 'tensors_epoch_%s_batch_%s_layer_%s.pt'%(epoch, batch, layer)))
            label = torch.load(os.path.join(root, 'tensors', 'labels_epoch_%s_batch_%s_layer_%s.pt' % (epoch, batch, layer)))
            xs.append(x)
            labels.append(label)

        # find min and max of the image over the epochs
        got_min = 1 * np.ones(xs[0].shape[1])
        got_max = -1 * np.ones(xs[0].shape[1])
        for x in xs:
            for channel in range(x.shape[1]):
                img = x[j, :, :]
                img = img[channel, :, :]
                get_min = np.min(np.array(img.cpu().detach())) - 0.01
                get_max = np.max(np.array(img.cpu().detach())) + 0.01
                if get_min <= got_min[channel]:
                    got_min[channel] = get_min
                if get_max >= got_max[channel]:
                    got_max[channel] = get_max

        #got_min, got_max = -0.2, 0.2

        print(got_min, got_max)

        # now print the images with appropriate scale
        for epoch, x in enumerate(xs):
            img = x[j, :, :]
            # print("Shape before layer", img.shape)
            channels = img.shape[0]
            h, w = img.shape[1], img.shape[2]
            plt.figure()
            f, axarr = plt.subplots(channels, 1, figsize=(4, 12))

            for channel in range(channels):
                if got_min[channel] == got_max[channel]:
                    got_max[channel] = got_max[channel] + 0.1

                img_chan = img[channel, :, :]
                img_chan = torch.transpose(img_chan, 0, 1)
                # print("img_chan", img_chan.shape)

                im = axarr[channel].imshow(img_chan.cpu().detach(), cmap='gray', vmin=got_min[channel], vmax=got_max[channel], interpolation=None)
                f.suptitle("Sample: %s" % ("Malignant" if label[j] == 0 else "Benign"), fontsize=20)
                plt.colorbar(im, ax=axarr[channel])
                #f.subplots_adjust(right=0.8)
                #cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                #f.colorbar(im, ax=cbar_ax)
            #plt.savefig(os.path.join(root, 'images', 'image_epoch_%s_batch_%s_layer_%s_index_%s.png' % (epoch, batch, layer, j)))
            plt.savefig(os.path.join(root, 'images',
                                     '%s_sample_epoch_%s.png' % ("malignant" if label[j] == 0 else "benign", epoch)))

            plt.show()
            plt.close()

