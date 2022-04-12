#---- Run from terminal using: ----
#python recommend_images.py return_recommendation <path to input image> <path to reference images>

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import torch

import os
import sys

from PIL import Image
import requests
from io import BytesIO

#Step 1. Embed 
sys.path.insert(0, os.path.abspath('./distribution_clustering'))
from save_dataset_features import save_features_test


#-- Step 2. Import --
#create object of image file with index it is read by
def get_file_key(path):
    
    file_keys = []

    for i, file in enumerate(os.listdir(path)):
    
        file_name = str(file).replace('.jpeg', '')
    
        temp_key = np.array((i, file_name))
        file_keys.append(temp_key)
    
    file_keys_df = pd.DataFrame(file_keys, columns = ['idx', 'file_name']).set_index('idx')

    return file_keys_df

# translate input file name to index within dataset
def get_image(file_name, file_keys_df):
    return int(file_keys_df.loc[file_keys_df['file_name'] == file_name].index[0])


#-- Step 3. Distance -- 
#get distance for any single input embedding and array of embeds to compare
def get_distance(input_embed, comparison_embeds):
    import scipy.spatial

    #translate to same size array as comparison
    input_expanded = np.repeat(input_embed, len(comparison_embeds), axis = 0)
    
    cluster_dist = scipy.spatial.distance.cdist(input_expanded, comparison_embeds, 'sqeuclidean')
    cluster_dist = cluster_dist[0] #only take one since all are duplicate
    return cluster_dist


#-- Step 6. Detect if duplicate -- 
#check distance between single input embed and embeds of existing directory to detect duplicate
    #i.e: detect if any distance == 0
def image_duplicate(input_embed, embed_dir, image_dir, plot_dup_image = False):
    
    embed_distances = get_distance(input_embed, embed_dir)
    
    if min(embed_distances) == 0:
        print('Photo Exists')
        dup_idx = np.where(embed_distances == min(embed_distances))[0][0]
        
        if bool(plot_dup_image):
            plt.imshow((image_dir[dup_idx][0].detach().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            
    else:
        print('Photo does not exist yet')
        dup_idx = None
    
    return dup_idx


# -- Combine all steps into one function --
def return_recommendation(input_path, dir_path, plot_similar = True):
    
    #Step 1. Embed input object
    save_features_test(input_path)
    
    #Step 2. Extract objects from path
    input_embed = np.load(os.path.join(input_path, 'features.npy'))
    cluster_centers = np.load(os.path.join(dir_path, 'centers.npy'))
    embed_dir = np.load(os.path.join(dir_path, 'features.npy'))
    #load base images
    tc = transforms.Compose([
        transforms.Resize((160, 240)),
        transforms.ToTensor()              
    ])
    image_dir = datasets.ImageFolder(os.path.join(dir_path, 'samples'), transform = tc)
    #get file/index key
    file_key = get_file_key(os.path.join(dir_path, 'samples/pics'))
    
    #Step 3. Get distance to cluster centers
    cluster_dist = get_distance(input_embed, cluster_centers)
    
    #Step 4. Get closest cluster (add one to offset 0 start index)
    closest_cluster = cluster_dist.argsort()[0] + 1
    print(closest_cluster)
    #Step 5. Get other images from cluster
    similar_imgs = os.listdir(os.path.join(dir_path, 'clusters', str(closest_cluster)))
    
    #turn into int corresponding to embed index, get embeds
    similar_idx = []
    similar_embeds = []
    for image in similar_imgs:
        temp_idx = str(image).replace('.jpeg', '')

        image_idx = get_image(temp_idx, file_key)
        #append based on image idx
        
        similar_idx.append(int(temp_idx))
        similar_embeds.append(embed_dir[image_idx])
    print(similar_idx)
    #Step 6. Check for duplicates
    similar_distances = get_distance(input_embed, np.asarray(similar_embeds))
    if min(similar_distances) == 0:
        dup_idx = np.where(similar_distances == 0)[0][0]
   
        #drop duplicate image id
        similar_idx = np.delete(similar_idx, dup_idx)
   
        print("Photo exists")
    else:
        dup_idx = None
        print("Photo does not exist")
        
    #Step 7. Return images within cluster
    #if plot is true plot similar images
    if bool(plot_similar):
        fig, axes = plt.subplots(2, 3, figsize = (10, 5))
        
        # Imports PIL module 
        from PIL import Image
        # open method used to open different extension image file
        input_name = os.listdir(os.path.join(input_path, 'samples'))[0]
        im = Image.open(os.path.join(input_path, 'samples', input_name)) 
  
        axes[0,0].imshow(im)
        axes[0,0].title.set_text('Input')
        
        for i, ax in enumerate(axes.flat[1:]):
            im = Image.open(os.path.join(dir_path, 'samples/pics', similar_imgs[i]))
            
            ax.imshow(im)
        
        plt.savefig(os.path.join(input_path, 'recommendations.jpeg'))


#enable command line running
if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3])