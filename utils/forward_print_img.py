import matplotlib
# matplotlib.use('module://backend_interagg')
# matplotlib.use('Qt5Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


os.environ['no_proxy'] = '127.0.0.1'


def print_img(x, root, layer, do_3D=False, nb_images=1, label=None):

    if not os.path.exists(os.path.join(root, "images")):
        os.makedirs(os.path.join(root, "images"))
    i = 0
    name = layer + "_epoch_" + str(i)
    while True:
        try:
            os.mkdir(os.path.join(root, "images", name))
            break
        except:
            i += 1
            name = layer + "_epoch_" + str(i)

    for j in range(nb_images):
        if do_3D == False:
            img = x[j, :, :]
            #print("Shape before layer", img.shape)
            channels = img.shape[0]
            h, w = img.shape[1], img.shape[2]
            plt.figure()
            f, axarr = plt.subplots(channels, 1)

            # chose the scale
            get_min = -0.6
            get_max = 0.6
            #get_min = np.min(np.array(img.cpu().detach())) - 0.01
            #get_max = np.max(np.array(img.cpu().detach())) + 0.01
            if get_min == get_max:
                get_max = get_max + 0.1
            # print("min", get_min)
            # print("max", get_max)

            for channel in range(channels):
                img_chan = img[channel, :, :]
                #print("img_chan", img_chan.shape)

                try:
                    im = axarr[channel].imshow(img_chan.cpu().detach(), cmap='gray', vmin=get_min, vmax=get_max)
                    #axarr[0].set_title("Sample: %s" % ("Malignant" if label[j] == 0 else "Benign"))
                except:
                    im = axarr.imshow(img_chan.cpu().detach(), cmap='gray', vmin=get_min, vmax=get_max)
                    #axarr.set_title("Sample: %s" % ("Malignant" if label[j] == 0 else "Benign"))

                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im, cax=cbar_ax)
                plt.savefig(os.path.join(root, "images", '%s/image_%s.png' % (name, j)))
                # plt.show()
            plt.close()


        if do_3D == True:
            img = x[j, :, :, :]
            #print("Shape before layer", img.shape)
            channels = img.shape[0]
            h, w = img.shape[1], img.shape[2]
            depth = img.shape[3]

            for channel in range(channels):
                #print("channel", channel)
                img_chan = img[channel, :, :, :]
                #print("img_chan", img_chan.shape)

                get_min = np.min(np.array(img_chan.cpu().detach())) - 0.01
                get_max = np.max(np.array(img_chan.cpu().detach())) + 0.01
                if get_min == get_max:
                    get_max = get_max + 0.1
                #print("min", get_min)
                #print("max", get_max)

                plt.figure(figsize=(24, 24))
                if depth >= 8:
                    #print("depth >= 8, not printing the rest:", depth)
                    f, axarr = plt.subplots(8, 1)
                    plt.suptitle("Print channel %s out of %s total depth is 8 out of %s" % (
                        channel + 1, channels, depth), fontsize=8)
                else:
                    f, axarr = plt.subplots(depth, 1)
                    plt.suptitle("Print channel %s out of %s" %
                                 (channel + 1, channels), fontsize=8)

                for i in range(depth):
                    if i < 8:
                        #print(i)
                        img_small = img_chan[:, :, i]
                        #print("img_small", img_small.shape)
                        #print(img_small)
                        try:
                            im = axarr[i].imshow(
                                img_small.cpu().detach(), cmap='gray', vmin=get_min, vmax=get_max)
                        except:
                            im = axarr.imshow(
                                img_small.cpu().detach(), cmap='gray', vmin=get_min, vmax=get_max)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im, cax=cbar_ax)
                #plt.title("Sample: %s" % ("Malignant" if label[j] == 0 else "Benign"))
                plt.savefig(os.path.join(root, "images", '%s/channel_%s_image_%s.png' % (name, channel, j)))

                # plt.show()
                plt.close()

    plt.close('all')

def save_tensor(x, root, epoch, batch_idx, layer, label=None):
    if not os.path.exists(os.path.join(root, "tensors")):
        os.makedirs(os.path.join(root, "tensors"))
    save_dir = os.path.join(root, "tensors")
    torch.save(x, '%s/tensors_epoch_%s_batch_%s_layer_%s.pt'%(save_dir, epoch, batch_idx, layer))
    torch.save(label, '%s/labels_epoch_%s_batch_%s_layer_%s.pt'%(save_dir, epoch, batch_idx, layer))