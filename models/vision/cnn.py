from numpy import save
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict

from tqdm import trange

class Net(nn.Module):
    def __init__(self, img_size, latent_dim, feature_labels, dropout):
        super().__init__()
        self.latet_dim = latent_dim
        self.img_size = img_size

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(in_features = 8*8*128, out_features = latent_dim)
        self.fcbn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(in_features = latent_dim, out_features = feature_labels)

        self.sigmoid = nn.Sigmoid()
        self.dropout_rate = dropout

    def forward(self, s):
        #we apply the convolution layers, followed by batch normalisation, 
        #maxpool and relu x 3
        s = self.bn1(self.conv1(s))        # batch_size x 32 x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))     # batch_size x 32 x 32 x 32
        s = self.bn2(self.conv2(s))        # batch_size x 64 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))     # batch_size x 64 x 16 x 16
        s = self.bn3(self.conv3(s))        # batch_size x 128 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))     # batch_size x 128 x 8 x 8

        #flatten the output for each image
        s = s.view(-1, 8*8*128)  # batch_size x 8*8*128

        #apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), p=self.dropout_rate, training=self.training)    # batch_size x latent_dim
        s = self.fc2(s) 
        s = self.sigmoid(s)
        return s

class Utility(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        # Layer parameters
        hidden_dim = 64
        self.latent_dim = latent_dim # model input 
        self.utility_out = 1 # number of utility predictions 

        # Fully connected layers
        self.lin1 = nn.Linear(self.latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers for mean and variance
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.view((batch_size, -1))
        # Fully connected layers with ReLu activations
        x = th.relu(self.lin1(x))
        x = th.relu(self.lin2(x))
        out = th.flatten(self.out(x))
        #out = out.view(-1, 1).unbind(-1) 

        return out


            
class CNN(nn.Module):
    def __init__(self, utility_type="Malloy", img_size=64, latent_dim=10, feature_labels=27, **kwargs):
        super(CNN, self).__init__()
        self.net = Net(img_size, latent_dim, feature_labels, (kwargs['kwargs']['dropout_percent'] / 100))
        self.net.fcbn1.register_forward_hook(self.get_activation('fcbn1'))
        self.utility = Utility(latent_dim=latent_dim)
        self.activation = {}
        self.upsilon = kwargs['kwargs']['upsilon']

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        feature_labels = self.net(x)
        util_input = self.activation['fcbn1']
        utility = self.utility(util_input)

        return feature_labels, utility 

    def save(self, save_dir, filename=None):
        """
        Save a model and corresponding metadata.

        Parameters
        ----------
        model : nn.Module
            Model.

        directory : str
            Path to the directory where to save the data.

        metadata : dict
            Metadata to save.
        """
        device = next(self.net.parameters()).device
        self.net.cpu()
        self.utility.cpu()

        th.save(self.net.state_dict(), os.path.join(save_dir, 'net'))
        th.save(self.utility.state_dict(), os.path.join(save_dir, 'utility'))

        self.net.to(device)  # restore device
        self.utility.to(device)  # restore device
    
    def load(self, exp_dir, args):
        #net = Net(args.img_size, 2*args.latent_dim, 9) # 
        #net.fcbn1.register_forward_hook(self.get_activation('fcbn1'))
        self.net.load_state_dict(th.load(os.path.join(exp_dir, 'net')))
        self.utility.load_state_dict(th.load(os.path.join(exp_dir, 'utility')))
        self.net.eval()
        self.utility.eval()

        return self
    
    def _utility_loss(self, utilities, recon_utilities, util_loss="mse", storer=None):
        if (utilities is None or recon_utilities is None):
            loss = 0
        if(util_loss == "mse"):
            lf = nn.MSELoss()
            loss = lf(utilities, recon_utilities)
        elif(util_loss == "L1"):
            lf = nn.L1Loss()
            loss = lf(utilities, recon_utilities)
        else:
            loss = ValueError("Unkown Utility Loss: {}".format(util_loss))
        
        if storer is not None:
            storer['recon_loss'].append(loss.item())
        
        return loss 

    def loss(self, data, utilities, recon_utilities, feature_labels, recon_labels, training, storer):
        utility_loss = self._utility_loss(utilities, recon_utilities, util_loss="mse")
        loss = nn.CrossEntropyLoss() 
        cnn_loss = loss(recon_labels, feature_labels)
        #print("true: ", feature_labels[0][0], " predicted ", recon_labels[0][0])

        return cnn_loss + (self.upsilon * utility_loss)


TRAIN_LOSSES_LOGFILE = "train_losses.log"

class Trainer():
    def __init__(self, model,   optimizer,
                                device="cuda",
                                logger=None,
                                save_dir="./",
                                is_progress_bar=False):
        self.model = model
        self.optimizer = optimizer 
        self.device = device
        self.logger = logger 
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar

        if self.logger is not None: 
            self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
            self.logger.info("Training Device: {}".format(self.device))
        

    def __call__(self, data_loader,
                 utilities=None,
                 feature_labels=None,
                 epochs=100,
                 checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, utilities, feature_labels, storer, epoch)
            if self.logger is not None:  self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,
                                                                               mean_epoch_loss))
            if self.logger is not None: self.losses_logger.log(epoch, storer)

            if epoch % checkpoint_every == 0 and epoch != 0:
                self.model.save( self.save_dir, filename="model-{}.pt".format(epoch))

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        if self.logger is not None: self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

    def _train_epoch(self, data_loader, utilities, feature_labels, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, data in enumerate(data_loader):
                iter_loss = self._train_iteration(data, utilities, feature_labels, storer)
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, utilities, feature_labels, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)
        feature_labels = th.from_numpy(feature_labels).to(self.device)
        
        """from PIL import Image
        import numpy as np 
        stim = np.transpose(data[1].detach().numpy().astype(np.uint8) * 255, [1, 2, 0])
        #stim = (data[1].detach().numpy() * 255 ).astype(np.uint8) 
        print(" stim shape: ", stim.shape)
        im = Image.fromarray(stim)
        im.show()

        print(" data shape in train iteration: ", data.shape)"""
        recon_labels, recon_utilities = self.model(data)
        loss = self.model.loss(data, utilities, recon_utilities,
                            feature_labels, recon_labels,
                            self.model.training, storer)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)
# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)