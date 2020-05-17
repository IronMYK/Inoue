import cv2 as cv2
import numpy as np
import os 
import random
import torchvision.transforms.functional as TF

from PIL import Image, ImageFilter
from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, ColorJitter, RandomCrop, Compose
from torch.utils.data import Dataset


cloud = cv2.imread('cloud.png', cv2.IMREAD_GRAYSCALE)
cloud = cv2.resize(cloud[:500, :500], (50, 50)) / 255. - 0.1

class CustomDataset(Dataset):
    def __init__(self, train=True):
        self.datapath_lines = "../BIPED/edges/edge_maps/{}".format("train/rgbr/real" if train else "test/rgbr")
        self.datapath_imgs = "../BIPED/edges/imgs/{}".format("train/rgbr/real" if train else "test/rgbr")
        self.lines = os.listdir(self.datapath_lines)
        self.imgs = os.listdir(self.datapath_imgs)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.unnormalize = Compose([ Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
        
    def __getitem__(self, index):
        edge = pil_loader(os.path.join(os.getcwd(), self.datapath_lines, self.lines[index]))
        img = pil_loader(os.path.join(os.getcwd(), self.datapath_imgs, self.imgs[index]))
        
        # Random scaling/rotation
        scale = random.random() + 0.5
        angle = random.random() * 0.25
        edge = TF.affine(edge, angle=angle, scale=scale, translate = (0, 0), shear=0)
        img = TF.affine(img, angle=angle, scale=scale, translate = (0, 0), shear=0)

        # Random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            edge = TF.hflip(edge)
        
        # Random crop
        i, j, h, w = RandomCrop.get_params(
            edge, output_size=(384, 384))
        edge = TF.crop(edge, i, j, h, w)
        img = TF.crop(img, i, j, h, w)
        
        # Random brightness/contrast/sat/hue
        t_jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        img = t_jitter(img)        
        
        # Deterioration
        if random.random() > 0.1:
            edge_d = TF.to_tensor(CustomDataset.deteriorate(edge))
            edge = 1 - TF.to_tensor(edge) # inversion as BIPED images are white on black
        else:
            edge = 1 - TF.to_tensor(edge) 
            edge_d = edge.detach()

        img = TF.to_tensor(img)
        return self.normalize(img), edge, edge_d

    def __len__(self):
        return len(self.lines)

    @staticmethod
    def deteriorate(img):
        sigma = random.random()
        b = np.random.binomial(1, 0.75, 3)
        res = img.copy()
        if b[0]: 
            res =  img.filter(ImageFilter.GaussianBlur(radius = 2 * sigma)) # blur
        # inversion as the BIPED images are black
        res = 1 - np.array(res) / 255.
        if b[1]: 
            res = (res + sigma)/(1. + sigma) # fading
        if b[2]:
            CustomDataset.fill_img(res, sigma) # fill
        return Image.fromarray(np.uint8(res * 255), 'L')

    @staticmethod
    def fill(img, i, j, size, sigma):
        case = np.random.randint(0, 3)
        # fading
        if case == 0:
            img[i - size:i + size, j - size:j + size] = (img[i - size:i + size, j - size:j + size] + sigma) / (1 + sigma)
        # blur
        elif case == 1:
            img[i - size:i + size, j - size:j + size] = 1
        # cloud texture
        else:
            x, x2 = max(0, i - size), min(len(img) - 1, i + size) 
            y, y2 = max(0, j - size), min(img.shape[1] - 1, j + size)
            img[x:x2, y:y2] =  sigma *  img[x:x2, y:y2] + (1 - sigma) * cloud[:x2 - x, :y2 - y]

    @staticmethod
    def fill_img(img, sigma):
        big_holes, small_holes = np.random.randint(5, 10), np.random.randint(50, 100)
        for _ in range(big_holes):
            size = np.random.randint(9, 21)
            i, j = np.random.randint(0, len(img)), np.random.randint(0, len(img[0]))
            CustomDataset.fill(img, i, j, size, sigma)
        for _ in range(small_holes):
            size = np.random.randint(1, 9)
            i, j = np.random.randint(0, len(img)), np.random.randint(0, len(img[0]))
            CustomDataset.fill(img, i, j, size, sigma)
    

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
    return img
    
if __name__ == '__main__':
    d = CustomDataset()
    v, e, g = d.__getitem__(5)
    x = (g * 255).data.numpy()
    print("EDGE", np.mean(x))
    print(np.array(v))
    e =  TF.to_pil_image(e)
    g = TF.to_pil_image(g)
    e.show()
    g.show()