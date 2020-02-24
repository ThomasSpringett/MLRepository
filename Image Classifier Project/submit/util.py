#!/usr/bin/env python
# coding: utf-8
#TWS. 2/22/2020
# -------------------------------------------------------------------------#
#           Import Libraries                                               #
# -------------------------------------------------------------------------#
from importlib import reload
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
def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="ImageClassifier File Analyzer")

    # Point towards image for prediction
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='data directory that has train/test/validation directorys',
                        default = '/home/workspace/ImageClassifier/flowers', 
                        nargs = '?',
                        const = 1,
                        required=False)


    # Parse args
    args = parser.parse_args()
    
    return args

def file_survey(datadir): 
    # This function helps in understanding the file structure and count 
    
    # Define directories: 
    data_dir = datadir
    train_dir = traindir(data_dir) 
    validation_dir = validationdir(data_dir) 
    test_dir = testdir(data_dir)
    
    #Display folder and file structure 
    print ("\nTotal Files in training dir: ", sum([len(files) for r, d, files in os.walk(train_dir)]))
    print ("Total Files in validation dir: ", sum([len(files) for r, d, files in os.walk(validation_dir)]))
    print ("Total Files in testing dir: ", sum([len(files) for r, d, files in os.walk(test_dir)]))
    print ("\nCurrent Working Directory ", os.getcwd())
    print ("\nflowers sub-folders\n",os.listdir(data_dir ))
    print ("\nTypical Image Path: ",test_dir + '/43/image_02365.jpg')
    print ("\nTraining folder: First 5 sub-folders. ")
    print (os.listdir(train_dir)[:5])
    print ("\nTesting folder: - First 5 sub-folders.")
    print (os.listdir(test_dir)[:5])
    print ("\nValidation folder: First 5 sub-folders. ")
    print (os.listdir('flowers/valid/')[:5])
    print ("\nFirst 5 photos in flowers/test/28")
    print (os.listdir('flowers/test/28')[:5])


def data_loader(data_dir):
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


# TODO: Save the checkpoint 
def save_checkpoint(model, save_dir, testloader): 
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': model.name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    print ("\nCheckpoint Saved\n") 

    torch.save(checkpoint, 'imageclassifier_checkpoint.pth')
    
def load_checkpoint(checkpoint_path):

    checkpoint = torch.load(checkpoint_path) 
    

    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
       
    elif checkpoint['architecture'] == 'alexnet':
        model = models.alexnet(pretrained = True)
        
    elif checkpoint['architecture'] == 'resnet18' : 
        model = models.resnet18(pretrained=True)
   
    else: 
        print ("model needs to be added") 
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.name = checkpoint['architecture']
 
    print ("\ncheckpoint loaded\n")
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

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def traindir(data_dir):
    return  data_dir + '/train'
def validationdir(data_dir):
    return  data_dir + '/valid'
def testdir(data_dir):
    return  data_dir + '/test'

def device(dev): 
   if torch.cuda.is_available() and dev == 'gpu':
         return torch.device("cuda")
   else:
         return torch.device("cpu")

# =============================================================================
#                    Run Program
# =============================================================================
def main(): 
    args = arg_parser()
    print ("device: ",device)                          
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    file_survey(data_dir)

# =============================================================================
#                    Run Program
# =============================================================================
if __name__ == '__main__': 

    main()





