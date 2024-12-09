# Setup
Tested on Conda python 3.9, CUDA = 12.6 Environment:
```
conda install pytorch == 2.3.1
torchvision==0.18.1 torchaudio == 2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-learn
```

# Run Training
Training code is train.py:
Following code can run the code.
train.ipynb was used for initial preparation and individual code debuging.
'''
python train.py --hps (dataset_name)
'''

# Dataset
Dataset was not included in the submission.
We used CIFAR10, ImageNet32 (Subset), and Oxford 102 Flower datasets.
All datasets are public data.

# Model
Model structure code was implemented on vae.py and components are implemented on vae_block folder

