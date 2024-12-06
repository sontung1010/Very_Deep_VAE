import numpy as np
import os
import time
import pandas as pd

# importing torch modules
import torch
from torch.utils.data import DataLoader

# importing from another files
from data import set_up_data
from train_helpers import accumulate_stats, save_model, update_ema
from utils_train import check_nans, create_logger, load_optimizer
from visualization import create_images, get_displaying_data
from train_setup import set_up_hyperparams, load_vaes


def train_main(H, training_dataset, validation_dataset, preprocess_fn, vae, ema_vae):
    logger = create_logger("Training Logger", log_file='training_log.txt')
    
    visualization_number = 8
    images_visualization_original, images_visualization_processed = get_displaying_data(validation_dataset, preprocess_fn, visualization_number, logger)
    stats = []
    iters_since_starting = 0
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_optimizer(H, vae, logger)
    
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()

    for epoch in range(starting_epoch, H.num_epochs):
        print('\n\n')
        logger.info("Starting epoch" + str(epoch))
        epoch_start = time.time()

        ## Creating a sample image display set
        # =======================================================================
        if epoch % H.epochs_per_eval == 0:
            file_name =  f'{H.save_dir}/displaying_images_' + str(iterate) + '.png'
            create_images(H, ema_vae, images_visualization_original, images_visualization_processed, file_name, logger)

        ## Load data
        # =======================================================================
        traindata_loader = DataLoader(
            training_dataset,
            batch_size=H.n_batch,
            drop_last=True,
            pin_memory=True,
            shuffle=True
        )

        for x in traindata_loader:
            ## Starting single step
            # =======================================================================
            input_data, target_data = preprocess_fn(x)
            step_start = time.time()
            vae.zero_grad() # initialize gradient for back prop

            # forward and backward process implementation
            forward_result = vae.forward(input_data, target_data) # forward result form: {elbo, distortion, rate}
            forward_result['elbo'].backward() # using elbo loss fucntion
            grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item() # limiting gradient value to H.grad_clip

            # check for norm value in distortion and rate
            return_dict = check_nans(forward_result)
            forward_result.update(return_dict)
            skipped_updates = 1
            if forward_result['distortion_nans'] == 0 and forward_result['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
                optimizer.step()
                skipped_updates = 0
                update_ema(vae, ema_vae, H.ema_rate)

            step_end = time.time()
            step_duration = step_end - step_start
            forward_result.update(skipped_updates=skipped_updates, iter_time=step_duration, grad_norm=grad_norm)
            stats.append(forward_result)

            ## Updating the result
            # =======================================================================
            scheduler.step()
            if iterate % H.iters_per_print == 0:
                logger.info("type: train_loss, lr: " + str(scheduler.get_last_lr()[0]) + ", epoch: " +str(epoch)+", step: " + str(iterate) + str(**accumulate_stats(stats, H.iters_per_print)))
            
            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logger.info("type: train_loss, lr: " + str(scheduler.get_last_lr()[0]) + ", epoch: " +str(epoch)+", step: " + str(iterate) + str(**accumulate_stats(stats, H.iters_per_print)))
                    fp = os.path.join(H.save_dir, 'latest')
                    logger.info('Saving model ' + str(iterate) + " to " + str(fp))
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)
            
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            logger.info("1 Epoch Finished - duration:"+ str(epoch_duration))

        ## Validation
        # =======================================================================
        if epoch % H.epochs_per_eval == 0:
            valid_stats = validation_main(H, ema_vae, validation_dataset, preprocess_fn)
            logger.info("type: eval_loss, epoch: " + str(epoch) + ", step: " + str(iterate) + str(**valid_stats))


def validation_main(H, ema_vae, data_valid, preprocess_fn):
    print('\n\n')
    stats_valid = []
    ## Load data
    # =======================================================================
    validation_data_loader = DataLoader(data_valid, 
                             batch_size=H.n_batch, 
                             drop_last=True, 
                             pin_memory=True, 
                             shuffle=True) 

    for x in validation_data_loader:
        ## Starting validation step
        # =======================================================================
        input_image, target_image = preprocess_fn(x)
        with torch.no_grad() : stats = ema_vae.forward(input_image, target_image)
        keys = sorted(stats.keys())
        stats_output = {k: float(stats[k]) for k in keys}    
        stats_valid.append(stats_output)
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(
        n_batches=len(vals),
        filtered_elbo=np.mean(finites),
        **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]}
    )
    return stats


def test_main(H, ema_vae, data_test, preprocess_fn):
    print('\n\n')
    logger = create_logger("Testing Logger", log_file="test_log.txt")
    logger.info("Starting testing phase.")
    stats = validation_main(H, ema_vae, data_test, preprocess_fn)
    logger.info("Test results:")
    print('=' * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    logger.info("type: test_loss, "+ str(**stats))


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    if H.test_eval:
        test_main(H, ema_vae, data_valid_or_test, preprocess_fn)
    else:
        train_main(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae)


if __name__ == "__main__":
    main()
