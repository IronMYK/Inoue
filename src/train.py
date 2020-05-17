import argparse
import json
import random
import torch

import numpy as np
from data import CustomDataset
from losses import loss_w1
from models import Generator, Restorer
from utils import Logger

from torchvision import transforms
from torch.nn import Sequential, L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

with open('cfg/config.json') as config_file:
    config = json.load(config_file)

def train(visualize=True):
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    random.seed(config['seed'])
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    restorer = Restorer().to(device)
    params = list(generator.parameters()) + list(restorer.parameters())
    # joint optimization
    optimizer = Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])

    # Start from previous session
    if config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'])
        generator.load_state_dict(checkpoint["generator"])
        restorer.load_state_dict(checkpoint["restorer"])
        optimizer.load_state_dict(checkpoint["optimize"])
    
    generator.train()
    restorer.train()

    # loss parameters
    alpha = config['loss']['alpha']
    beta = config['loss']['beta']
    gamma = config['loss']['gamma']
    l1_loss = L1Loss().to(device)

    # Dataset
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)    

    # Training Loop
    print("Training begins")
    logger = Logger(visualize=visualize) # logs and visdom plots, images
    batch_count, checkpoint_count = 0, 0
    for epoch in range(config['n_epochs']):
        print("Starting Epoch #{}...".format(epoch))
        for batch_i, images in enumerate(dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            photos = images[0].to(device)
            true_sketches = images[1].to(device)
            deteriorated_sketches = images[2].to(device)

            generated_sketches = generator(photos)
            corrected_sketches = restorer(generated_sketches)
            corrected_deteriorated = restorer(deteriorated_sketches)
            
            loss_base = loss_w1(generated_sketches, true_sketches, gamma) # L_base
            loss_aux = loss_w1(corrected_sketches, true_sketches, gamma) # L_aux
            loss_res = l1_loss(corrected_deteriorated, true_sketches) # L_res
            loss_joint = loss_base + alpha * loss_aux + beta * loss_res # L_joint
            loss_joint.backward()
            optimizer.step()

            # logs batch losses 
            logger.log_iteration(epoch, batch_i, 
                loss_base.item(), 
                loss_res.item(), 
                loss_aux.item(), 
                loss_joint.item()
            )

            if visualize and batch_count % config['sample_interval'] == 0:
                logger.draw( 
                    corrected_sketches[0].data.numpy() * 255, 
                    true_sketches[0].data.numpy() * 255,
                    corrected_deteriorated[0].data.numpy() * 255, 
                    dataset.unnormalize(photos[0]).data.numpy()
                ) # draws a sample on Visdom 

            # checkpoint save
            if batch_count % config['save_checkpoint_interval'] == 0:
                torch.save({
                    "generator": generator.state_dict(), 
                    "restorer": restorer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                    }, "{}/checkpoint_{}.pth".format(config['checkpoint_dir'], checkpoint_count))
                checkpoint_count += 1
            batch_count += 1

        if visualize:
            logger.plot_epoch(epoch) # plots the average losses on Visdom


parser = argparse.ArgumentParser(description="Line Drawing training")
parser.add_argument('--v', default=True, action='store_false', help="to not visualize")
args = parser.parse_args()

if __name__ == "__main__":  
    train(visualize=args.v)

