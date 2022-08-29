import argparse
import logging
import sys
import os
import copy 
from configparser import ConfigParser

import warnings
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

import math, random 
from models.learning.frl import Feature_RL, frl_env

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]
FEATURE_MAP = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]

def get_latent(stimuli, model, train_loader):
    stim_indices = []
    for stim in stimuli: 
        for feature_index, features  in enumerate(FEATURE_MAP):
            if(all(np.array(stim)-1==features)):
                index=feature_index
        stim_indices.append(index)
    #print("stim indicies: ", stim_indices)
    for _, data in enumerate(train_loader):
        data_subset = []
        for stim_index in stim_indices:
            stim = torch.unsqueeze(data[stim_index], 0) 
            #im = np.transpose((data[stim_index].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
            #im = Image.fromarray(im)
            #im.show()
            recon_batch, latent_dist, latent_sample, recon_utilities = model(stim)

            return latent_dist, latent_sample

def predict_utilities(stimuli, model, train_loader, inv_temp):
    stim_indices = []
    utilities = []
    for stim in stimuli: 
        for feature_index, features  in enumerate(FEATURE_MAP):
            if(all(np.array(stim)-1==features)):
                index=feature_index
        stim_indices.append(index)
    #print("stim indicies: ", stim_indices)
    for _, data in enumerate(train_loader):
        data_subset = []
        for stim_index in stim_indices:
            stim = torch.unsqueeze(data[stim_index], 0) 
            #im = np.transpose((data[stim_index].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
            #im = Image.fromarray(im)
            #im.show()
            _, _, _, util = model(stim)
            utilities.append(util.detach().numpy())
    
    utilities = np.array(utilities).clip(min=0) + 1e-6
    values = np.array([math.exp(inv_temp * utilities[stim_index]) for stim_index in range(3)])
    value_softmax = [values[0] / sum(values), values[1] / sum(values), values[2] / sum(values)]

    #print(" Utilities: ", utilities)
    #print(" Values: ", values)
    
    guess = np.random.choice([0,1,2], 1, p=[value_softmax[0], value_softmax[1], value_softmax[2]])[0]
    prediction = stimuli[guess]

    return (prediction, guess, [value_softmax[0], value_softmax[1], value_softmax[2]])

def get_updated_utilities(features):
    updated_utilities = np.array([np.sum((features[feature[0]] , features[feature[1] + 3], features[feature[2] + 6])) for feature in FEATURE_MAP])
    return (torch.from_numpy(updated_utilities)).float() 

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
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
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

from statistics import NormalDist
def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

def kl_normal_loss(pm, plv, qm, qlv):
    qv = np.exp(qlv)
    pv = np.exp(plv)
    pm = np.asarray(pm)
    qm = np.asarray(qm)
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N

def calculateKLD(means, logvars, cor_idx):
    similarUtilityKLD = []
    diffUtilityKLD = []

    for stimuli_group1, (stimuli_means1, stimuli_logvars1) in enumerate(zip(means, logvars)):
        for stimuli_group2, (stimuli_means2, stimuli_logvars2) in enumerate(zip(means, logvars)):
            if(stimuli_group1 == cor_idx and stimuli_group2 != cor_idx): continue # different utility 
            if(stimuli_group1 != cor_idx and stimuli_group2 == cor_idx): continue # different utility 
            for stimuli_index1, (latent_means1, latent_logvars1) in enumerate(zip(stimuli_means1, stimuli_logvars1)):
                for stimuli_index2, (latent_means2, latent_logvars2) in enumerate(zip(stimuli_means2, stimuli_logvars2)):
                    kld = kl_normal_loss(latent_means1, latent_logvars1, latent_means2, latent_logvars2)
                    similarUtilityKLD.append(kld)
    
    for stimuli_group1, (stimuli_means1, stimuli_logvars1) in enumerate(zip(means, logvars)):
        for stimuli_group2, (stimuli_means2, stimuli_logvars2) in enumerate(zip(means, logvars)):
            if(stimuli_group1 == cor_idx and stimuli_group2 == cor_idx): continue # same utility 
            if(stimuli_group1 != cor_idx and stimuli_group2 != cor_idx): continue # same utility 
            for stimuli_index1, (latent_means1, latent_logvars1) in enumerate(zip(stimuli_means1, stimuli_logvars1)):
                for stimuli_index2, (latent_means2, latent_logvars2) in enumerate(zip(stimuli_means2, stimuli_logvars2)):
                    kld = kl_normal_loss(latent_means1, latent_logvars1, latent_means2, latent_logvars2)
                    diffUtilityKLD.append(kld)

    return np.mean(similarUtilityKLD), np.mean(diffUtilityKLD)

"""
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
return mean + std * eps
"""
def calculateOverlapsHL(means, logvars, cor_idx):
    highUtilityOverlaps = []
    lowUtilityOverlaps = []

    for stimuli_group1, (stimuli_means1, stimuli_logvars1) in enumerate(zip(means, logvars)):
        for stimuli_group2, (stimuli_means2, stimuli_logvars2) in enumerate(zip(means, logvars)):
            if(stimuli_group1 != cor_idx or stimuli_group2 != cor_idx): continue # low utility 
            for stimuli_index1, (latent_means1, latent_logvars1) in enumerate(zip(stimuli_means1, stimuli_logvars1)):
                for stimuli_index2, (latent_means2, latent_logvars2) in enumerate(zip(stimuli_means2, stimuli_logvars2)):
                    if(stimuli_group1 == stimuli_group2 and stimuli_index1 == stimuli_index2): continue # Exact same stimuli 
                    for latent_mean1, latent_logvar1, latent_mean2, latent_logvar2, in zip(latent_means1, latent_logvars1, latent_means2, latent_logvars2):
                        overlap = NormalDist(mu=latent_mean1, sigma= np.exp(latent_logvar1 * 0.5)).overlap(NormalDist(mu=latent_mean2, sigma= np.exp(latent_logvar2 * 0.5)))
                        highUtilityOverlaps.append(overlap)
    
    for stimuli_group1, (stimuli_means1, stimuli_logvars1) in enumerate(zip(means, logvars)):
        for stimuli_group2, (stimuli_means2, stimuli_logvars2) in enumerate(zip(means, logvars)):
            if(stimuli_group1 == cor_idx or stimuli_group2 == cor_idx): continue # high utility 
            for stimuli_index1, (latent_means1, latent_logvars1) in enumerate(zip(stimuli_means1, stimuli_logvars1)):
                for stimuli_index2, (latent_means2, latent_logvars2) in enumerate(zip(stimuli_means2, stimuli_logvars2)):
                    if(stimuli_group1 == stimuli_group2 and stimuli_index1 == stimuli_index2): continue # Exact same stimuli 
                    for latent_mean1, latent_logvar1, latent_mean2, latent_logvar2, in zip(latent_means1, latent_logvars1, latent_means2, latent_logvars2):
                        overlap = NormalDist(mu=latent_mean1, sigma= np.exp(latent_logvar1 * 0.5)).overlap(NormalDist(mu=latent_mean2, sigma= np.exp(latent_logvar2 * 0.5)))
                        lowUtilityOverlaps.append(overlap)

    return np.mean(highUtilityOverlaps), np.mean(lowUtilityOverlaps)

def calculateOverlaps(means, logvars, cor_idx):
    sameUtilityOverlaps = []
    differentUtilityOverlaps = []

    for stimuli_group1, (stimuli_means1, stimuli_logvars1) in enumerate(zip(means, logvars)):
        for stimuli_group2, (stimuli_means2, stimuli_logvars2) in enumerate(zip(means, logvars)):
            if(stimuli_group1 == cor_idx and stimuli_group2 != cor_idx): continue # different utility 
            if(stimuli_group1 != cor_idx and stimuli_group2 == cor_idx): continue # different utility 
            for stimuli_index1, (latent_means1, latent_logvars1) in enumerate(zip(stimuli_means1, stimuli_logvars1)):
                for stimuli_index2, (latent_means2, latent_logvars2) in enumerate(zip(stimuli_means2, stimuli_logvars2)):
                    if(stimuli_group1 == stimuli_group2 and stimuli_index1 == stimuli_index2): continue # Exact same stimuli 
                    for latent_mean1, latent_logvar1, latent_mean2, latent_logvar2, in zip(latent_means1, latent_logvars1, latent_means2, latent_logvars2):
                        overlap = NormalDist(mu=latent_mean1, sigma= np.exp(latent_logvar1) ** -2).overlap(NormalDist(mu=latent_mean2, sigma= np.exp(latent_logvar2) ** -2))
                        sameUtilityOverlaps.append(overlap)
    
    for stimuli_group1, (stimuli_means1, stimuli_logvars1) in enumerate(zip(means, logvars)):
        for stimuli_group2, (stimuli_means2, stimuli_logvars2) in enumerate(zip(means, logvars)):
            if(stimuli_group1 == cor_idx and stimuli_group2 == cor_idx): continue # same utility 
            if(stimuli_group1 != cor_idx and stimuli_group2 != cor_idx): continue # same utility 
            for stimuli_index1, (latent_means1, latent_logvars1) in enumerate(zip(stimuli_means1, stimuli_logvars1)):
                for stimuli_index2, (latent_means2, latent_logvars2) in enumerate(zip(stimuli_means2, stimuli_logvars2)):
                    if(stimuli_group1 == stimuli_group2 and stimuli_index1 == stimuli_index2): continue # Exact same stimuli 
                    for latent_mean1, latent_logvar1, latent_mean2, latent_logvar2, in zip(latent_means1, latent_logvars1, latent_means2, latent_logvars2):
                        overlap = NormalDist(mu=latent_mean1, sigma= np.exp(latent_logvar1) ** -2).overlap(NormalDist(mu=latent_mean2, sigma= np.exp(latent_logvar2) ** -2))
                        differentUtilityOverlaps.append(overlap)

    return np.mean(sameUtilityOverlaps), np.mean(differentUtilityOverlaps)

def get_utility(x, model):
    util_input = torch.from_numpy(np.array(x).reshape(2,3)).unsqueeze(0).float()
    utility = model.utility(util_input).detach().numpy()
    return -1 * utility # using scipy minimize, want to maximize utility 

def main(args):
    args.img_size = get_img_size(args.dataset)
    mat = io.loadmat('./data/niv/responses/BehavioralDataOnline.mat')
    """
    Choices - choices (800 trials x 3 features x 22 subjects), NaN for missed trials 
    Outcomes - outcomes, 1 or 0 (800 trials x 22 subjects), NaN for missed trials
    Stimuli - stimuli (800 trials x 3 dimensions x 3 features x 22 subjects; the first dimension is colors, then shapes, then textures)
    RelevantDim - relevant dimension in each trial (800 trials x 22 subjects) 
    CorrectFeature - correct feature in each trial (800 trials x 22 subjects)
    ReactionTimes - reaction times (800 trials x 22 subjects), NaN for missed trials
    """
    data = mat["DimTaskData"][0,0]
    Choices = mat['DimTaskData'][0][0]['Choices']
    Outcomes = mat['DimTaskData'][0][0]['Outcomes']
    Stimuli = mat['DimTaskData'][0][0]['Stimuli']
    RelevantDim = mat['DimTaskData'][0][0]['RelevantDim']
    CorrectFeature = mat['DimTaskData'][0][0]['CorrectFeature']
    ReactionTimes = mat['DimTaskData'][0][0]['ReactionTimes']
    Feature_Thetas_500, Feature_Thetas_300 = Feature_RL.get_thetas()

    # retrain and predict 
    ResponseAccuracy = pd.DataFrame()
    RepresentationOverlap = pd.DataFrame()
    states = np.matrix([
    [1,2,3],
    [1,2,3],
    [1,2,3]
    ])

    actions = np.linspace(0,1,num=1)
    env = frl_env(states=states, actions=actions, rewards=[])

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
    exp_dir = os.path.join(RES_DIR, args.name)
    if(not os.path.exists(exp_dir)): os.mkdir(exp_dir)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    
    if(not args.is_eval_only):
        # pretrain model or load pretrained on reconstructing stimuli set
        model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
        model = model.to(device)  # make sure trainer and viz on same device
        #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger)
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                            device=device,
                            logger=logger,
                            save_dir=exp_dir,
                            is_progress_bar=not args.no_progress_bar)
        utilities = np.ones((27))* 0.5
        utilities = torch.from_numpy(utilities.astype(np.float64)).float()
        trainer(train_loader,
                utilities=utilities, 
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))
    
    model = load_model(exp_dir, is_gpu=not args.no_cuda)
    model.to(device)
    base_model = copy.deepcopy(model)

    train_loader = get_dataloaders(args.dataset, batch_size=args.batch_size)
    # 0,22 standard full participant range 
    for participant_id in range(0,22):
        alpha_300 = Feature_Thetas_300[participant_id][0] 
        beta_300 = Feature_Thetas_300[participant_id][1] 
        delta_300 = Feature_Thetas_300[participant_id][2] 

        alpha_500 = Feature_Thetas_500[participant_id][0] 
        beta_500 = Feature_Thetas_500[participant_id][1] 
        delta_500 = Feature_Thetas_500[participant_id][2] 

        last_prediction = -1
        old_relevant = -1
        trial_game_index = 0
        game_index = 0

        factorization_parameters = np.ones(args.latent_dim)

        for trial_num in range(0,800):
            if(trial_num == 500): # reset when trial type switches
                model = copy.deepcopy(base_model)
            color_feature_1 = Stimuli[trial_num][0][0][participant_id]
            shape_feature_1 = Stimuli[trial_num][0][1][participant_id]
            texture_feature_1 = Stimuli[trial_num][0][2][participant_id] 
            color_feature_2 = Stimuli[trial_num][1][0][participant_id]
            shape_feature_2 = Stimuli[trial_num][1][1][participant_id] 
            texture_feature_2 = Stimuli[trial_num][1][2][participant_id] 
            color_feature_3 = Stimuli[trial_num][2][0][participant_id] 
            shape_feature_3 = Stimuli[trial_num][2][1][participant_id] 
            texture_feature_3 = Stimuli[trial_num][2][2][participant_id] 

            relevant = RelevantDim[trial_num][participant_id]
            correct = CorrectFeature[trial_num][participant_id]
            outcome = Outcomes[trial_num][participant_id]

            if(math.isnan(Choices[trial_num][0][participant_id])):
                continue 

            chosen = [
                int(Choices[trial_num][0][participant_id]),
                int(Choices[trial_num][1][participant_id]),
                int(Choices[trial_num][2][participant_id])
            ]
            
            stimulus = [
                [color_feature_1, shape_feature_1, texture_feature_1],
                [color_feature_2, shape_feature_2, texture_feature_2],
                [color_feature_3, shape_feature_3, texture_feature_3]
            ]

            for option_index, option in enumerate(stimulus):
                if (option == chosen): choice_index = option_index
            
            if(relevant != old_relevant):
                trial_game_index = 0
                game_index += 1
                # Resetting model  negatively impacts predictive accuracy
                # model = copy.deepcopy(base_model)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                loss_f = get_loss_f(args.loss,
                                        n_data=len(train_loader.dataset),
                                        device=device,
                                        **vars(args))
                trainer = Trainer(model, optimizer, loss_f,
                                        device=device,
                                        logger=None,
                                        save_dir=exp_dir,
                                        is_progress_bar=False)
                base_utilities = np.ones((27))* 0.5
                base_utilities = torch.from_numpy(base_utilities.astype(np.float64)).float()
                epochs = args.model_epochs 
                trainer(train_loader,
                    utilities=base_utilities, 
                    epochs=epochs, 
                    checkpoint_every=10000)
            
            
            inv_temp = beta_300 if trial_num >= 500 else beta_500
            (prediction, guess, bvae_percentages) = predict_utilities(stimulus, model, train_loader, inv_temp=inv_temp)

            latent = get_latent(stimulus, model, train_loader)
            means, logvars = latent[0]
            sample = latent[1]
            
            means = means.detach().numpy()[0]
            logvars = logvars.detach().numpy()[0]
            sample = sample.detach().numpy()[0]

            # shape = 7, color = 9, textr = 6

            #print(means)
            #print(logvars)
            print(sample)

            print(np.sum(sample * factorization_parameters))

            # update factorization parameters: 

            print(outcome)

            # outcome
            
            assert(False)
            

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


