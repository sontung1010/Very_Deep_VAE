import numpy as np
import os
import time
import pandas as pd

# importing torch modules
import torch
from torch.utils.data import DataLoader

# importing from another files
from data import prepare_data
from utils_train import check_nans, create_logger, load_optimizer, stats_batch_processing, saving_model, update_ema, add_row_train, add_row_val
from visualization import create_images, create_images_test, get_displaying_data, formatting_text, formatting_text_validation
from train_setup import hyperparameter_setting, load_model_custom


def train_main(H, training_dataset, validation_dataset, preprocess_fn, vae, ema_vae, logger):
    columns_train = ['epoch', 'step', 'iteration_time', 'elbo', 'elbo_filtered', 'skipped_updates']
    df_train = pd.DataFrame(columns=columns_train)
    df_train.to_csv(H.save_dir + '/saving_train_stats.csv')
    columns_val = ['epoch', 'step', 'iteration_time', 'elbo', 'elbo_filtered']
    df_val = pd.DataFrame(columns=columns_val)
    df_val.to_csv(H.save_dir + '/saving_validation_stats.csv')
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
            file_name =  f'{H.save_dir}/displaying_images_' + str(epoch) + '.png'
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
            x = [x]
            ## Starting single step
            # =======================================================================
            input_data, target_data = preprocess_fn(x)
            #print(input_data.shape, target_data.shape)
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
            forward_result_numpy = {
                key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
                for key, value in forward_result.items()
            }
            stats.append(forward_result_numpy)
            # logger.debug('stats:'+str(stats))

            ## Updating the result
            # =======================================================================
            scheduler.step()
            if iterate % H.iters_per_print == 0:
                accumulated = stats_batch_processing(stats, H.iters_per_print)
                format_text = formatting_text(accumulated)
                logger.info("type: train_loss, lr: " + str(scheduler.get_last_lr()[0]) + ", epoch: " +str(epoch)+", step: " + str(iterate) + format_text)
                # ['epoch', 'setp', 'iteration_time', 'elbo', 'elbo_filtered', 'skipped_updates']
                df_train = add_row_train(df_train, epoch, iterate, accumulated['iter_time'], accumulated['elbo'], accumulated['elbo_filtered'], accumulated['skipped_updates'])
            
            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    accumulated = stats_batch_processing(stats, H.iters_per_print)
                    format_text = formatting_text(accumulated)                    
                    logger.info("type: train_loss, lr: " + str(scheduler.get_last_lr()[0]) + ", epoch: " +str(epoch)+", step: " + str(iterate) + format_text)
                    fp = os.path.join(H.save_dir, 'latest')
                    logger.info('Saving model ' + str(iterate) + " to " + str(fp))
                    saving_model(epoch, fp, vae, ema_vae, optimizer, H)              
            
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        logger.info("1 Epoch Finished - duration:"+ str(epoch_duration))

        ## Validation
        # =======================================================================
        if epoch % H.epochs_per_eval == 0:
            valid_stats = validation_main(H, ema_vae, validation_dataset, preprocess_fn, logger)
            valid_stats_text = formatting_text_validation(valid_stats)
            logger.info("type: eval_loss, epoch: " + str(epoch) + ", step: " + str(iterate) + str(valid_stats_text))

            ## saving information
            # =======================================================================
            saving_model(epoch, os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)
            df_train.to_csv(H.save_dir + '/saving_train_stats.csv')
            df_val = add_row_val(df_val, epoch, iterate, valid_stats['elbo'], valid_stats['filtered_elbo'])
            df_val.to_csv(H.save_dir + '/saving_validation_stats.csv')


def validation_main(H, ema_vae, data_valid, preprocess_fn, logger):
    print('\n\n')
    logger.info("Intiating validation.")
    stats_valid = []
    ## Load data
    # =======================================================================
    validation_data_loader = DataLoader(data_valid, 
                             batch_size=H.n_batch, 
                             drop_last=True, 
                             pin_memory=True, 
                             shuffle=True) 

    for x in validation_data_loader:
        x = [x]
        ## Starting validation step
        # =======================================================================
        input_image, target_image = preprocess_fn(x)
        with torch.no_grad() : forward_result = ema_vae.forward(input_image, target_image)
        forward_result_numpy = {
            key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
            for key, value in forward_result.items()
        }
        stats_valid.append(forward_result_numpy)
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(
        n_batches=len(vals),
        filtered_elbo=np.mean(finites),
        **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]}
    )
    return stats


def test_main(H, ema_vae, data_test, preprocess_fn, logger):
    print('\n\n')
    logger.info("Starting testing phase.")
    images_visualization_original, images_visualization_processed = get_displaying_data(data_test, preprocess_fn, 4, logger)
    file_name =  f'{H.save_dir}/displaying_images_test.png'
    create_images_test(H, ema_vae, images_visualization_original, images_visualization_processed, file_name, logger)
    stats = validation_main(H, ema_vae, data_test, preprocess_fn, logger)
    logger.info("Test results:")
    logger.info('=' * 50)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    test_stats_text = formatting_text_validation(stats)
    logger.info("type: test_loss, "+ str(test_stats_text))


def main():
    logger = create_logger("Training Logger", log_file='training_log.txt')
    Parameters = hyperparameter_setting(logger)
    Parameters, data_train, data_validation_test, preprocess_fn = prepare_data(Parameters)
    vae, ema_vae = load_model_custom(Parameters, logger)
    if Parameters.test_eval:
        test_main(Parameters, ema_vae, data_validation_test, preprocess_fn, logger)
    else:
        train_main(Parameters, data_train, data_validation_test, preprocess_fn, vae, ema_vae, logger)


if __name__ == "__main__":
    main()
