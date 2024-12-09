import numpy as np
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

def get_displaying_data(data, preprocess_fn, num, logger):
    dataloader = DataLoader(data, batch_size=num)
    x = next(iter(dataloader))
    original_image = x
    logger.debug(f"Original image shape: {original_image.shape}")
    preprocessed_image = preprocess_fn([x])[0]
    logger.debug(f"Preprocessed image shape: {preprocessed_image.shape}")
    return original_image, preprocessed_image

def plt_plotting(images, filename):
    column_names = ['Original', 'Layer 0-6', 'Layer 0-12', 'Layer 0-18', 'Layer 0-24',
                    'Layer 0-30', 'Layer 0-36']
    rcParams['font.family'] = 'Times New Roman'
    fig1, axes = plt.subplots(nrows=1, ncols=7, figsize=(14, 3))

    for j in range(7): 
        image_here = images[j,1,:,:,:]
        axes[j].imshow(image_here)
        axes[j].axis("off")
        axes[j].set_title(column_names[j], fontsize=20) 

    plt.savefig(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.9]) 
    plt.show()
    return 

def unconditional_plotting(images, filename):
    fig2, axes2 = plt.subplots(nrows=1, ncols=7, figsize=(14, 3))

    for j in range(7): 
        image_here = images[j,0,:,:,:]
        axes2[j].imshow(image_here)
        axes2[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.9]) 
    plt.savefig(filename)
    plt.show()
    return 


def create_images_test(H, ema_vae, viz_batch_original, viz_batch_processed, fname, logger):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    #print(np.shape(batches))
    mb = viz_batch_processed.shape[0]
    
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        #print("printing i:", i)
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], temperature=0.1))
    #print(np.shape(batches))
    #print(type(batches))
    plt_plotting(np.array(batches), fname)

    unconditional = []
    for i in range(14):
        unconditional.append(ema_vae.forward_unconditional_samples(mb, temperature=1))
    unconditional_plotting(np.array(unconditional), fname[:-4] + '_uncond.png')


def create_images(H, ema_vae, viz_batch_original, viz_batch_processed, fname, logger):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], temperature=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_unconditional_samples(mb, temperature=t))
    
    # Combine all batches into a single image
    n_rows = len(batches)
    im = (
        np.concatenate(batches, axis=0)
        .reshape((n_rows, mb, *viz_batch_processed.shape[1:]))
        .transpose([0, 2, 1, 3, 4])
        .reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3])
    )
    
    im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    logger.info('Printing samples to ' + str(fname))
    cv2.imwrite(fname, im_bgr)

def formatting_text(stats):
    try:
        iteration_time = ' iteration_time: ' + str(stats['iter_time']) + ' '
        elbo_str = 'elbo: ' + str(stats['elbo'].item()) + ' '
        elbo_filtered_str = 'elbo_filtered: ' + str(stats['elbo_filtered'].item()) + ' '
        rate_nans = 'rate_nans: ' + str(stats['rate_nans'].item()) + ' '
        skipped_updates = 'skipped_updates: ' + str(stats['skipped_updates']) + ' '
    except : return "Failed logging."
    return iteration_time + elbo_str + elbo_filtered_str + rate_nans + skipped_updates

def formatting_text_validation(stats):
    try:
        n_batches_str = ' n_batches: ' + str(stats['n_batches']) + ' '
        elbo_str = 'elbo: ' + str(stats['elbo']) + ' '
        elbo_filtered_str = 'elbo_filtered: ' + str(stats['filtered_elbo']) + ' '
        rate_str = 'rate: ' + str(stats['rate']) + ' '
        distortion_str = 'distortion: ' + str(stats['distortion']) + ' '
    except : return "Failed logging."
    return n_batches_str + elbo_str + elbo_filtered_str + rate_str + distortion_str

def plot_loss_function(path_training, path_validation):
    rcParams['font.family'] = 'Times New Roman'
    csv_training = pd.read_csv(path_training)
    csv_validation = pd.read_csv(path_validation)
    training_step = csv_training['step']
    training_elbo = csv_training['elbo_filtered']
    validation_step = csv_validation['step']
    validation_elbo = csv_validation['elbo_filtered']
    
    plt.figure(figsize=(7, 4))
    plt.plot(training_step, training_elbo, label='Training Loss', linewidth=2)
    plt.plot(validation_step, validation_elbo, label='Validation Loss', linewidth=2)
    plt.xlabel("Iteration", fontsize=16)  
    plt.ylabel("ELBO (negative value)", fontsize=16) 
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid()
    plt.title("Training and Validation Loss", fontsize=18)
    plt.tight_layout()
    plt.show()

## This code is for generating figure for report.
# plot_loss_function('./saved_models/dec07_cifar/saving_train_stats.csv','./saved_models/dec07_cifar/fixed_cifar_loss_log.csv')