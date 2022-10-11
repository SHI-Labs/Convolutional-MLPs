'''
This script is adapted from https://github.com/yitu-opensource/T2T-ViT/blob/main/visualization_vit.ipynb
You may define a new main_{model} to visualize the specific model to your interests
'''
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from timm.models.mlp_mixer import mixer_b16_224
from timm.models.resnet import resnet50d
from timm.models import create_model, resume_checkpoint, convert_splitbn_model

import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch Visualize')
parser.add_argument('--model', metavar='MODEL', default='vit',
                    help='path to dataset')

def main_mixer():
    model = create_model(
            'mixer_b16_224',
            pretrained=True,
            num_classes=1000)

    block_weights = [] 
    block_layers = [] 
    model_children = list(model.children())

    all_block = [] 
    for model_child in list(model.children()):
        sub_children = list(model_child.children())
        for sub_child in sub_children:
            all_block.append(sub_child)

    # read and visualize an image
    img = cv2.imread("images/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    # pass the image through all the layers
    results = []
    layers = [2,5,8,11]
    for i in range(12):
        # pass the result from the last layer to the next layer
        if i == 2:
            img = img.flatten(2).transpose(1, 2)
        img = all_block[i](img)
        if i in layers:
            results.append(img)

    stages = 4 
    for num_stage in range(stages): 
        plt.figure(figsize=(16, 12))
        layer_viz = results[num_stage]
        layer_viz = layer_viz.data
        layer_viz = layer_viz.squeeze(0).transpose(0, 1).reshape(768, 14, 14)
        for i, filter in enumerate(layer_viz):
            if i == 12:
                break
            plt.subplot(3, 4, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_stage} feature maps...")
        plt.savefig(f"output/vis/mixer/stage_{num_stage}.png")  # change the path to save the feature maps
        plt.close()

def main_resnet():
    model = create_model(
            'resnet50d',
            pretrained=False,
            num_classes=1000)

    block_weights = [] 
    block_layers = [] 
    model_children = list(model.children())

    all_block = []
    for model_child in list(model.children()):
        if type(model_child) != torch.nn.modules.container.ModuleList:
            all_block.append(model_child)
        else:
            sub_children = list(model_child.children())
            for sub_child in sub_children:
                all_block.append(sub_child)

    # read and visualize an image
    img = cv2.imread("images/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    # pass the image through all the layers
    results = []
    layers = [0,4,5,6,7]
    for i in range(len(all_block)):
        # pass the result from the last layer to the next layer
        img = all_block[i](img)
        if i in layers:
            results.append(img)

    stages = 5   # change the number of length to control how many layers you want to visualize
    for num_stage in range(stages): 
        plt.figure(figsize=(16, 12))
        layer_viz = results[num_stage]
        layer_viz = layer_viz.data
        layer_viz = layer_viz.squeeze(0)
        for i, filter in enumerate(layer_viz):
            if i == 12:
                break
            plt.subplot(3, 4, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_stage} feature maps...")
        plt.savefig(f"output/vis/resnet/stage_{num_stage}.png")  # change the path to save the feature maps
        plt.close()

if __name__ == '__main__':
    args = parser.parse_args()
    if args.model == 'mixer':
        main_mixer()
    elif args.model == 'resnet':
        main_resnet()