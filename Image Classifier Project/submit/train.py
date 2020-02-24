#!/usr/bin/env python
# coding: utf-8
# TWS 2/22/2020
# =========================================================================#
#            Import Libraries.                                             #
# =========================================================================#
import torch
import argparse
import json
import util
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import time 

# =========================================================================#
#           Function Definitions                                           #
# =========================================================================#


def args_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="ImageClassifier File Analyzer")

    # Point towards image for prediction
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Checkpoint Directory',
                        default = '/home/workspace/ImageClassifier/flowers', 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='Data folder with /train, /test and /valid subfolders',
                        default = '/home/workspace/ImageClassifier/flowers', 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--arch', 
                        type=str, 
                        help='torch vision model architecture',
                        default = 'vgg16', 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--lr', 
                        type=float, 
                        help='learning rate ',
                        default = 0.001, 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--hidden', 
                        type=int, 
                        help='hidden units',
                        default = 4096, 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--epochs', 
                        type=int, 
                        help='number of units ',
                        default = 1, 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--gpu',
                        type=str, 
                        help='if gpu  is to be used',
                        default = 'gpu', 
                        nargs = '?',
                        const = 1,
                        required=False)


    # Parse args
 
    args = parser.parse_args()
    
    print (args)
    
    return args

#Creating untrained feed-forward network 

def classifier_model(architecture, hidden_units, lr):
    
    number_input_features = 4096 #default 
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        number_input_features = model.classifier[0].in_features
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
        number_input_features = alexnet.classifier[6].in_features;
    elif architecture == 'resnet18' : 
        model_ft = models.resnet18(pretrained=True)
        number_input_features= model_ft.fc.in_features
    else:
        print("Sorry but {} is not a valid model.Please add model to code".format(pretrained_model))
        
    # Freezing parameters (turn off gradients) to prevent backpropagation
    #Gradients will not be calculated and updated. 

    for param in model.parameters():
        param.requires_grad = False
        '''
    if type(hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
        print("Number of Hidden Layers specificed as 4096.")
    '''
    from collections import OrderedDict

    #Define new classifier 
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(number_input_features,hidden_units)),
        ('relu',nn.ReLU()),
        ("dropout", nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units,102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    #attached new classifier to model

    model.classifier = classifier
    model.name = architecture 

    criterion  = nn.NLLLoss() 

    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model ,optimizer ,criterion 

def validation(model, validation_loader, criterion,device): 
    test_loss = 0
    accuracy = 0 
    
    for ii, (inputs, labels) in enumerate(validation_loader): 
        if ii == len(validation_loader) - 1:
            print ("Number of Validation Batches run: ", ii)

        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item() 
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.cuda.FloatTensor).mean() 
    return test_loss, accuracy 

def training_model(trainloader, testloader, validation_loader,model, criterion,optimizer,epochs, device):
    print ("\nTraining Start")
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
                    valid_loss, accuracy = validation(model, validation_loader,criterion,device)
                
  
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
    
def testing_model(model,testloader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('\nAccuracy achieved by the network on test images is: %d%%' % (100 * correct / total))
    



# =============================================================================#
#                                  Main                                        #
# =============================================================================#

def main(): 
 
    args = args_parser()
    device = util.device(args.gpu) 
    print ("\nDevice being used: ", device)
   
    save_dir = args.save_dir 
                                 
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        # Look at this dictionary to understand the data:
        #print (next(iter(data_dir/cat_to_name.items())))
                
    trainloader, validation_loader, testloader = util.data_loader(args.data_dir)
    
    model, optimizer,criterion = classifier_model(args.arch, args.hidden, args.lr)
    
    model.to(device) 
    
    training_model(trainloader,testloader,validation_loader,model, criterion, optimizer, args.epochs, device)
    
    testing_model(model,testloader,device)
    
    util.save_checkpoint(model, save_dir, testloader)

# =============================================================================#
#                    Run Program                                               #
# =============================================================================#
if __name__ == '__main__': 

    main()

