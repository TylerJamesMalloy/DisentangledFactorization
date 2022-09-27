import argparse
import logging
import sys
import os
import copy 
from configparser import ConfigParser

from collections import defaultdict

import warnings
warnings.filterwarnings('error')
warnings.simplefilter(action='ignore', category=FutureWarning)


from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import UTIL_LOSSES, LOSSES, RECON_DIST, get_loss_f, _reconstruction_loss
from disvae.models.vae import MODELS, UTILTIIES 
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining

from torch.utils.data import Dataset, DataLoader

import torch 
from torch import optim
import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 

import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T

import math, random 
from models.learning.frl import Feature_RL, frl_env
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

test_images = [[],[],[],[]]

for images in os.listdir('./data/celebb/both'):
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        # display
        test_images[0].append(images)
for images in os.listdir('./data/celebb/glasses'):
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        # display
        test_images[1].append(images)
for images in os.listdir('./data/celebb/hats'):
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        # display
        test_images[2].append(images)
for images in os.listdir('./data/celebb/neither'):
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        # display
        test_images[3].append(images)


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-ut', '--utility-type',
                       default=default_config['utility'], choices=UTILTIIES,
                       help='Type of utility prediction model to use.')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-ul', '--util-loss',
                       default=default_config['util_loss'], choices=UTIL_LOSSES,
                       help="Type of Utility loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    utility = parser.add_argument_group('BetaH specific parameters')
    utility.add_argument('-u', '--upsilon', type=float,
                       default=default_config['upsilon'],
                       help="Weight of the utility loss parameter.")
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Eval options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')
    
    modelling = parser.add_argument_group('L&DM modelling specific options')
    modelling.add_argument('--model-epochs', type=int,
                            default=default_config['model_epochs'],
                            help='Number of epochs to train utility prediction model.')
    modelling.add_argument('--trial-update', type=str,
                            default=default_config['trial_update'],
                            help='Source for util predictions.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

def main(args):
    args.img_size = get_img_size(args.dataset)

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    
    device = get_device(is_gpu=not args.no_cuda)

    loss_f = get_loss_f(args.loss,
                        device=device,
                        **vars(args))

    data = pd.DataFrame()

    latent_dims = [100]

    #for name_idx, name in enumerate(["btcvae_celeba_a-5"]):
    for name_idx, name in enumerate(["btcvae_celeba_best_ld100"]):
        latent_dim = latent_dims[name_idx]
        setattr(args, 'latent_dim', latent_dim)
        setattr(args, 'loss', "betaH")

        #exp_dir = os.path.join(RES_DIR, args.name)
        exp_dir = os.path.join(RES_DIR, name)
        #if(not os.path.exists(exp_dir)): os.mkdir(exp_dir)
        logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

        #model = load_model(exp_dir, is_gpu=not args.no_cuda)
        model = load_model(exp_dir + "/bandit_1K/")

        image_folders = ['./data/celebb/both/','./data/celebb/glasses/','./data/celebb/hats/','./data/celebb/neither/']

        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger)

        #for group_idx, image_group in enumerate(test_images): 
        #    folder = image_folders[group_idx]
        #    for image_path in image_group: 


        #image = read_image(folder + image_path).unsqueeze(0).float()
        #recon, latent_dist, latent_sample, utility = model(image)
        #print("utility: ", utility)
        #print(_reconstruction_loss(image, recon, distribution="gaussian"))

        # sunglasses 310 (1, 54)  
        # both 1096 (4, 72)
        # neither 1220 (4,196)

        for idx, data in enumerate(train_loader):
            if (idx == 1):  
                # hat lady
                im = np.transpose((data[0][58].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()

                im.save("recons/hat_original.jpg")

                recon, latent_dist, latent_sample, utility = model(data[0][58].unsqueeze(0))

                im = np.transpose((recon[0].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()

                im.save("recons/hat_ld100_bandit.jpg")

                im = np.transpose((data[0][60].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()

                im.save("recons/glasses_original.jpg")

                # sunglasses lady
                recon, latent_dist, latent_sample, utility = model(data[0][60].unsqueeze(0))

                im = np.transpose((recon[0].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()
                im.save("recons/glasses_ld100_bandit.jpg")
            
            if (idx == 4):  
                # both 
                im = np.transpose((data[0][31].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()

                im.save("recons/both_original.jpg")

                recon, latent_dist, latent_sample, utility = model(data[0][31].unsqueeze(0))

                im = np.transpose((recon[0].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()
                im.save("recons/both_ld100_bandit.jpg")

                # neither
                im = np.transpose((data[0][155].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()

                im.save("recons/neither_original.jpg")

                recon, latent_dist, latent_sample, utility = model(data[0][155].unsqueeze(0))

                im = np.transpose((recon[0].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
                im = Image.fromarray(im)
                #im.show()
                im.save("recons/neither_ld100_bandit.jpg")
            
            if (idx > 4): continue 

        
#  python .\bandit.py btcvae_celeba_ld5_b256

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)



