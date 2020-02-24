
#TWS. 2/22/2020 
# =============================================================================#
#                                  Import Libraries                            #
# =============================================================================#
import util
import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from torchvision import models

# =============================================================================#
#                                  Define Functions                            #
# =============================================================================#

def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="ImageClassifier File Analyzer")

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='data directory that has train/test/validation directorys',
                        default = '/home/workspace/ImageClassifier/flowers/test/43/image_02329.jpg', 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--category_names', 
                        type=str, 
                        help='category names for classifier',
                        default = 'cat_to_name.json',
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Number of top categories',
                        default = 5,
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--gpu',
                        type=str, 
                        help=' gpu to be used',
                        default = 'gpu', 
                        nargs = '?',
                        const = 1,
                        required=False)
    parser.add_argument('--load_dir', 
                        type=str, 
                        help='Checkpoint Load Directory',
                        default = '/home/workspace/ImageClassifier/imageclassifier_checkpoint.pth', 
                        nargs = '?',
                        const = 1,
                        required=False)


    # Parse args
    args = parser.parse_args()
    
    return args

def predict(image_path, model, device, cat_to_name, top_k ):
    # Predict the class (or classes) of an image using a trained deep learning model.
             
    model.to(device)  
    print ("device in predict: ", device)
    
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(util.process_image(image_path), 
                                                 axis=0)).type(torch.FloatTensor).to(device)

    # Probabilities by passing through the function 
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


def myprint(flowers,probs): 
    flowers_probs = dict(zip(flowers, probs * 100))
    i = 0 
    print ("Top Flowers")
    for flower,prob in flowers_probs.items():
      i += 1 
      print ("Rank {}: ".format(i) + flower + ",  Probability {}% ".format(ceil(prob)))

 
# =============================================================================#
#                                  Main                                        #
# =============================================================================#

def main(): 
    args = arg_parser()
    device = util.device(args.gpu) 
    print ("device: ",device)
    print ("load_dir is: ",args.load_dir)
 
    
    # Load model trained with train.py
    model = util.load_checkpoint(args.load_dir)
 
             
    with open(args.category_names, 'r') as f:
      cat_to_name = json.load(f)
      # Look at this dictionary to understand the data:
      #print (next(iter(cat_to_name.items())))

    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict('/home/workspace/ImageClassifier/flowers/test/43/image_02329.jpg', model, device, cat_to_name, args.top_k)
    
    myprint(top_flowers,top_probs)

# =============================================================================#
#                    Run Program                                               #
# =============================================================================#
if __name__ == '__main__': 

    main()