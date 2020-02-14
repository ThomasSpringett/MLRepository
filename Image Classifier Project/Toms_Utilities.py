#!/usr/bin/env python
# coding: utf-8



# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


#Setup directory structure 

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Method to display folder and file structure 
def file_survey(): 
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

    global test_image 
    test_image = mpimg.imread("flowers/test/28/image_05242.jpg", "r")
    plt.imshow(test_image)
    print('-' * 20)

file_survey()




def data_loader():
 #Define ransforms for the training, validation, and testing sets
 train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

 test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



 #Load the datasets with ImageFolder
 train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
 validation_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
 test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

 # TODO: Using the image datasets and the trainforms, define the dataloaders

 trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
 validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64)
 testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
 print ("trainloader count: ",len(trainloader))
 print ("validation_loader count: ",len(validation_loader))
 print ("testloader count: ",len(testloader))
 return trainloader, validation_loader,testloader 


# In[6]:
trainloader, validation_loader, testloader = data_loader()

#method to show images 
def imshow1(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(trainloader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow1(out, title=None)



import json

#create dictionary of catagorey names 

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    # Look at this dictionary to understand the data:
    print (next(iter(cat_to_name.items())))

 

# # Process
# The transfer learning process will be done as follows:
# <ol>
#     <li> Get pre-trained model from PyTorch</li>
#     <li> Create Classifier</li>
#     <li> Turn off feature parameters from being updated </li>
#     <li> Train classifier using feedforward through model </li>
#     <li> Define loss model and optimizer (for classifier parameters only) </li>
# 

# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[8]:


#Building  and training the network
model = models.vgg16(pretrained=True)
#model


# In[9]:


#Creating untrained feed-forward network 

def classifier_model(pretrained_model, lr = 0.001):
    
    if pretrained_model == 'vgg16':
        model = models.vgg16(pretrained=True) 
    else:
        print("Im sorry but {} is not a valid model.Please add model to code".format(pretrained_model))
        



    # Freezing parameters (turn off gradients) to prevent backpropagation
    #Gradients will not be calculated and updated. 

    for param in model.parameters():
        param.requires_grad = False
    
    from collections import OrderedDict

    #Define new classifier 
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088,4096)),
        ('relu',nn.ReLU()),
        ("dropout", nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(4096,102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    #attached new classifier to model

    model.classifier = classifier
    #model

    criterion  = nn.NLLLoss() 

    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    
    return model ,optimizer ,criterion 

    #Next section will train the classifier


# In[10]:


model, optimizer,criterion = classifier_model('vgg16')


# In[11]:


#determine if GPU can be used

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print ("Processor:",device)


# In[12]:


def training_model(model, criterion,optimizer,epochs, device):
    print ("Training Start")
    since = time.time() 
    step = 0
    validate_every = 25
    
    for epoch in range(epochs): 
        running_loss = 0
        
        print('-' * 10)
    
        #load the data and 
        for ii, (inputs, labels) in enumerate(trainloader):

            if ii == len(trainloader) - 1:
                print ("\nNumber of Training Batches run: ", ii)
            step += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            #Training the category classifier: with forward pass 
            optimizer.zero_grad() 
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()
        
            #Finished Training 
        
            if step % validate_every == 0:
                model.eval() 
            
                #Validation Loop 
                with torch.no_grad(): 
                    valid_loss, accuracy = validation(model, validation_loader,criterion)
                
  
                print("Epoch: {}/{} | ".format(epoch+1, epochs),
                  "Training Loss: {:.4f} | ".format(running_loss/validate_every),
                  "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                  "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
       
            running_loss = 0
            model.train()
       
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print("\nTraining process is now complete!!") 


# In[ ]:


training_model(model, criterion, optimizer, 5, device)





# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[ ]:


# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx
checkpoint = {'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}

torch.save(checkpoint, 'imageclassifier_checkpoint.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[ ]:


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint():
    """
    Loads deep learning model checkpoint.
    """
    
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


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[ ]:


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

imshow(process_image('flowers/test/28/image_05242.jpg'))


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[ ]:


def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    def predict(image_path, model, top_k=5):
    
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
    
    # No need for GPU 
    model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def sanity_check():

    # Define image path
    #image_path = test_dir + '/28/image_05242.jpg'
    #image_path = test_dir + '/28/image_05214.jpg'
    #image_path = test_dir + '/28/image_05277.jpg'
    image_path = test_dir + '/28/image_05253.jpg'
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)

    # Set up title
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);

    # Make prediction
    probs, labs, flowers = predict(image_path, model) 

    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()
    print ("Finished")

sanity_check()





