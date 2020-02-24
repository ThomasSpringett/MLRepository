#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------------#
#           Import Libraries                                               #
# -------------------------------------------------------------------------#

import time 
import torch
import os
import torchvision
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image
import PIL
import seaborn as sns
import json
import argparse 
# =========================================================================#
#           Function Definitions                                           #
# =========================================================================#


def file_survey(): 
    # Function to display folder and file structure 
    print ("Total Files in training dir: ", sum([len(files) for r, d, files in os.walk('flowers/train/')]))
    print ("Total Files in validation dir: ", sum([len(files) for r, d, files in os.walk('flowers/valid/')]))
    print ("Total Files in testing dir: ", sum([len(files) for r, d, files in os.walk('flowers/test/')]))
    print ("\nCurrent Working Directory ", os.getcwd())
    print ("\nflowers sub-folders\n",os.listdir('flowers'))
    print ("\nTypical Image Path: ",test_dir + "/43/image_02365.jpg")
    print ("\nTraining folder: First 5 sub-folders. ")
    print (os.listdir('flowers/train/')[:5])
    print ("\nTesting folder: - First 5 sub-folders.")
    print (os.listdir('flowers/test/')[:5])
    print ("\nValidation folder: First 5 sub-folders. ")
    print (os.listdir('flowers/valid/')[:5])
    print ("\nFirst 5 photos in flowers/test/28")
    print (os.listdir('flowers/test/28')[:5])


def data_loader():
 #Define ransforms for the training, validation, and testing sets
 global train_data
 train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
 train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
 trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
 test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

 test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
 validation_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

 validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
 testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
                                         
 print ("trainloader: ",len(trainloader))
 print ("validationloader: ",len(validationloader))
 print ("testloader: ",len(testloader))
        
 return trainloader, validationloader, testloader 

def save_checkpoint(model):
    # Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}
    torch.save(checkpoint, 'imageclassifier_checkpoint.pth')


def load_checkpoint():
    # Load the saved file
    checkpoint = torch.load("imageclassifier_checkpoint.pth")
    
    # Download pretrained model
    model = models.vgg16(pretrained=True);
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model
       
def process_image(image):
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor

# =============================================================================
#                    Run Program
# =============================================================================
def main(): 
    
        
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
                                          
    with open('cat_to_name.json', 'r') as f:
     cat_to_name = json.load(f)
     # Look at this dictionary to understand the data:
     print (next(iter(cat_to_name.items())))
              
    def device():
       return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device())
    print ("Processor:",device())
    

# =============================================================================
#                    Run Program
# =============================================================================
if __name__ == '__main__': main()





