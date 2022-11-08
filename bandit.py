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
from disvae.models.losses import UTIL_LOSSES, LOSSES, RECON_DIST, get_loss_f
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


def utility(latent_dim, hypothesis):
    utility = 0

    for idx, dim in enumerate(latent_dim):
        #utility += (1 / len(latent_dim)) * (dim.detach().numpy() * (hypothesis[idx]))
        utility += (dim * (hypothesis[idx]))

    return utility

def softmax(utilities, tau):
    try:
        distribution = np.exp(utilities * tau) / np.sum(np.exp(utilities * tau))
    except Exception as e: 
        deterministic = np.zeros_like(utilities)
        max_idx = np.argmax(utilities)
        deterministic[max_idx] = 1
        return deterministic
    
    if np.isnan(distribution).any():
        deterministic = np.zeros_like(utilities)
        max_idx = np.argmax(utilities)
        deterministic[max_idx] = 1
        return deterministic
    
    return distribution

def update_hypotheses(hypotheses_space, chosen_latent_sample, reward):
    lr = 1.0 # Deterministic reward setting 

    feature_map = [] # Assume 1 or 2 features relevant 
    for i in range(args.latent_dim):
        single_feature_hypothesis = np.zeros(args.latent_dim)
        single_feature_hypothesis[i] = 1
        
        for j in range(args.latent_dim):
            if i == j: continue 
            double_feature_hypothesis = copy.copy(single_feature_hypothesis)
            double_feature_hypothesis[j] = 1
            feature_map.append(double_feature_hypothesis)
        
        feature_map.append(single_feature_hypothesis)

    new_hypotheses_space = []
    new_hypotheses_ranks = []
    for idx, hypothesis in enumerate(hypotheses_space):
        utility_prediction = utility(chosen_latent_sample, hypothesis)
        features = feature_map[idx]

        reward_prediction_error = np.abs(utility_prediction - reward)
        new_hypotheses_ranks.append(reward_prediction_error)

        new_hypothesis = []
        num_features = np.sum(features)

        for feature_idx, feature in enumerate(features):
            if(feature == 0): 
                new_hypothesis.append(0)
                continue 
            new_parameter = ((1/num_features) * np.abs((utility_prediction - reward))) / chosen_latent_sample[feature_idx]
            new_parameter = np.clip(new_parameter, 1e-8, 1e8)
            new_hypothesis.append(new_parameter)

        new_hypotheses_space.append(new_hypothesis)
        
        #print(utility(chosen_latent_sample, new_hypothesis) == reward)

    return new_hypotheses_space, new_hypotheses_ranks


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

    data = pd.DataFrame()

    latent_dims = [100]

    #for name_idx, name in enumerate(["btcvae_celeba_ld5_b256", "btcvae_celeba_ld10_b256", "btcvae_celeba_ld25_b256", "btcvae_celeba_ld50_b256", "btcvae_celeba_ld100_b256"]):
    for name_idx, name in enumerate(["btcvae_celeba_best_ld100"]):
        latent_dim = latent_dims[name_idx]
        setattr(args, 'latent_dim', latent_dim)
        setattr(args, 'loss', "betaH")

        #exp_dir = os.path.join(RES_DIR, args.name)
        exp_dir = os.path.join(RES_DIR, name)
        #if(not os.path.exists(exp_dir)): os.mkdir(exp_dir)
        logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
        
        device = get_device(is_gpu=not args.no_cuda)
        
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        model.eval()

        image_folders = ['./data/celebb/both/','./data/celebb/glasses/','./data/celebb/hats/','./data/celebb/neither/']

        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger)

        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                    **vars(args))

        """

        trainer = Trainer(model, optimizer, loss_f,
                    device=device,
                    logger=logger,
                    save_dir=exp_dir,
                    is_progress_bar=not args.no_progress_bar)
        """

        left_image_folder = image_folders[0]
        right_image_folder = image_folders[1]

        left_image_idx = np.random.choice(range(25))
        right_image_idx = np.random.choice(range(25))

        left_image_path = left_image_folder + test_images[0][left_image_idx]
        right_image_path = right_image_folder + test_images[1][right_image_idx]

        left_image = read_image(left_image_path).unsqueeze(0).float()
        right_image = read_image(right_image_path).unsqueeze(0).float()

        recon, _, _, _ = model(left_image)

        im = np.transpose((recon[0].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
        im = Image.fromarray(im)
        im.show()

        recon, _, _, _ = model(right_image)

        im = np.transpose((recon[0].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
        im = Image.fromarray(im)
        im.show()

        assert(False)

        import datetime 
        for batch in range(1):
            hypotheses_space = np.zeros((args.latent_dim**2, args.latent_dim)) # assume 2 features relevant 
            hypotheses_ranks = np.zeros((args.latent_dim**2))
            ## Create hypothesis space for possible fits (linear functions only)
            for episode in range(1000):
                if(episode % 1000 == 0): 
                    print(episode)
                    print(datetime.datetime.now())
                image_types = [0,1,2,3]
                left_image_type = np.random.choice(image_types)
                image_types.remove(left_image_type)
                right_image_type = np.random.choice(image_types)

                image_rewards = [75, 50, 25, 0]

                ## Get two images from different folders 
                left_image_folder = image_folders[left_image_type]
                right_image_folder = image_folders[right_image_type]

                left_image_idx = np.random.choice(range(25))
                right_image_idx = np.random.choice(range(25))

                left_image_path = left_image_folder + test_images[left_image_type][left_image_idx]
                right_image_path = right_image_folder + test_images[right_image_type][right_image_idx]

                left_image = read_image(left_image_path).unsqueeze(0).float()
                right_image = read_image(right_image_path).unsqueeze(0).float()

                ## Get latent features of two images 
                _, _, left_latent_sample, left_utility_pred = model(left_image)
                _, _, right_latent_sample, right_utility_pred = model(right_image)

                left_latent_sample = left_latent_sample.detach().numpy()[0]
                right_latent_sample = right_latent_sample.detach().numpy()[0]

                left_utility_pred = left_utility_pred.detach().numpy()[0]
                right_utility_pred = right_utility_pred.detach().numpy()[0]

                best_hypothesis = np.argmin(hypotheses_ranks)
                hypothesis = hypotheses_space[best_hypothesis]

                left_utility  = utility(left_latent_sample , hypothesis)
                right_utility = utility(right_latent_sample, hypothesis)
                utilities = np.array([left_utility, right_utility])

                dist = softmax(utilities, 10)
                choice = np.random.choice([0,1], p=dist)
                unchosen = 1 if choice == 0 else 0
                chosen_type = left_image_type if choice == 0 else right_image_type
                chosen_latent_sample = left_latent_sample if choice == 0 else right_latent_sample
                reward = image_rewards[chosen_type]
                
                true_utility = [image_rewards[left_image_type], image_rewards[right_image_type]]

                correct = 1 if true_utility[choice] >= true_utility[unchosen] else 0 
                predictive_accuracy = dist[0] if utilities[0] >= utilities[1] else dist[1]

                chosen_image = left_image if choice == 0 else right_image
                storer = defaultdict(list)
                #chosen_image = chosen_image.unsqueeze(0)
                reward_tensor = np.array([reward])
                reward_tensor = torch.from_numpy(reward_tensor.astype(np.float64)).float()

                #trainer._train_iteration(chosen_image, reward_tensor, storer, 0)
                
                for _ in range(100):
                    recon_batch, latent_dist, latent_sample, recon_reward = model(chosen_image)
                    loss = loss_f(chosen_image, recon_batch, reward_tensor, recon_reward,
                                        latent_dist, True,
                                        storer, latent_sample=latent_sample)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #print(loss.item())

                hypotheses_space, hypotheses_ranks = update_hypotheses(hypotheses_space, chosen_latent_sample, reward)
                utility_prediction_error = ((left_utility_pred - image_rewards[left_image_type]) ** 2) + ((right_utility_pred - image_rewards[left_image_type]) ** 2)

                #episode = 10000 * int(episode/10000)
                d = {"Episode":episode, "Correct": correct, "Reward": reward, "Predictive Accuracy": predictive_accuracy, "Latent Size": len(left_latent_sample), "Utility Prediction Error": utility_prediction_error}

                data = data.append(d, ignore_index=True)

            save_model(model, exp_dir + "/bandit/", metadata=vars(args))

        data.to_pickle("./representation_results_best_ld100.pkl")

    ax = sns.lineplot(x="Episode", y="Utility Prediction Error", hue="Latent Size", data=data)
    plt.title("Model Percent of Correct")
    plt.show()

    # plot, save data 
    
    print("Got to end of training")

#  python .\bandit.py btcvae_celeba_ld5_b256

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)



