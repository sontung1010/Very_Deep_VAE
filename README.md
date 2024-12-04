
# environment
```bash
# cuda 12.6, gpu RTX 4070
conda create -n virtual_vdvae python = 3.9

```

# dependancy
```bash
# following modules need to be installed.
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install imageio
conda install -c conda-forge mpi4py
pip install scikit-learn
```

# prepare the dataset
manually download one of the dataset and unzip in the same directory.


# run the training
after moving working directory, and prepare the dataset.
```bash
python train.py --hps cifar10
```

# Dataset preparation
### ImageNet 32
```bash
# 119M parameter model, trained for 1.7M iters (about 2.5 weeks on 32 V100)
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/imagenet32-iter-1700000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/imagenet32-iter-1700000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/imagenet32-iter-1700000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/imagenet32-iter-1700000-opt.th
python train.py --hps imagenet32 --restore_path imagenet32-iter-1700000-model.th --restore_ema_path imagenet32-iter-1700000-model-ema.th --restore_log_path imagenet32-iter-1700000-log.jsonl --restore_optimizer_path imagenet32-iter-1700000-opt.th --test_eval
# should give 2.6364 nats per dim, which is 3.80 bpd
```

### ImageNet 64
```bash
# 125M parameter model, trained for 1.6M iters (about 2.5 weeks on 32 V100)
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
python train.py --hps imagenet64 --restore_path imagenet64-iter-1600000-model.th --restore_ema_path imagenet64-iter-1600000-model-ema.th --restore_log_path imagenet64-iter-1600000-log.jsonl --restore_optimizer_path imagenet64-iter-1600000-opt.th --test_eval
# should be 2.44 nats, or 3.52 bits per dim
```

### FFHQ-256
```bash
# 115M parameters, trained for 1.7M iterations (or about 2.5 weeks) on 32 V100
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-opt.th
python train.py --hps ffhq256 --restore_path ffhq256-iter-1700000-model.th --restore_ema_path ffhq256-iter-1700000-model-ema.th --restore_log_path ffhq256-iter-1700000-log.jsonl --restore_optimizer_path ffhq256-iter-1700000-opt.th --test_eval
# should be 0.4232 nats, or 0.61 bits per dim
```

### FFHQ-1024
```bash
# 115M parameters, trained for 1.7M iterations (or about 2.5 weeks) on 32 V100
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq1024-iter-1700000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq1024-iter-1700000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq1024-iter-1700000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq1024-iter-1700000-opt.th
python train.py --hps ffhq1024 --restore_path ffhq1024-iter-1700000-model.th --restore_ema_path ffhq1024-iter-1700000-model-ema.th --restore_log_path ffhq1024-iter-1700000-log.jsonl --restore_optimizer_path ffhq1024-iter-1700000-opt.th --test_eval
# should be 1.678 nats, or 2.42 bits per dim
```

### CIFAR-10
```bash
# 39M parameters, trained for ~1M iterations with early stopping (a little less than a week on 2 GPUs)
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/cifar10-seed0-iter-900000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/cifar10-seed1-iter-1050000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/cifar10-seed2-iter-650000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/cifar10-seed3-iter-1050000-model-ema.th
python train.py --hps cifar10 --restore_ema_path cifar10-seed0-iter-900000-model-ema.th --test_eval
python train.py --hps cifar10 --restore_ema_path cifar10-seed1-iter-1050000-model-ema.th --test_eval
python train.py --hps cifar10 --restore_ema_path cifar10-seed2-iter-650000-model-ema.th --test_eval
python train.py --hps cifar10 --restore_ema_path cifar10-seed3-iter-1050000-model-ema.th --test_eval
# seeds 0, 1, 2, 3 should give 2.879, 2.842, 2.898, 2.864 bits per dim, for an average of 2.87 bits per dim.
```
