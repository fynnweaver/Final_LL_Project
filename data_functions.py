def select_photos(tfdf, ratio = (3, 2), ratio_tol = 0.25, filter_range = 5000):
    """Selects photos of a similar aspect ratio.
    
    INPUT
    ----
    
    OUTPUT
    -----
    
    """
    
    from math import isclose
    import tensorflow_datasets as tfds
    
    images = []
    labels = []
    
    comparison = ratio[0] / ratio[1]
    

    for (image, label) in tfds.as_numpy(tfdf.take(filter_range)):
        current_ratio = image.shape[1] / image.shape[0]
        #only take if match dim object
        if isclose(comparison, current_ratio, abs_tol = ratio_tol):
            images.append(image)
            labels.append(label)
                
    return images, labels


def tfds_to_frame(tfds_images, tfds_labels, subsample = False):
    """Converts tensor frame dataset images and labels into a dataframe.
    
    INPUT
    ----
    
    OUTPUT
    -----
    
    """
    import pandas as pd
    
    #convert enumerated images list to dataframe
    image_index = list(enumerate(tfds_images))
    index_df = pd.DataFrame(image_index, columns = ['index', 'image']).set_index('index')

    #convert labels to dataframe
    labels_df = pd.DataFrame(tfds_labels, columns = ['label'])
 
    #concat into key
    key = pd.concat([labels_df, index_df], axis = 1)
    
    if index_df.shape[0] != labels_df.shape[0] or labels_df.shape[0] != key.shape[0]:
        print('Warning: dataframe row length changed!')
    
    if bool(subsample):
        return key.iloc[:subsample]
    else:
        return key

    
def sub_sample_domain(cat_df, sample_total = 200, random = True, category = True):
    """Randomly creates subsample of every category in cat_df with count above the sample_total threshold.
    
    INPUT
    ----
    
    OUTPUT
    -----
     
    """
    if bool(category):
        #save categories containing less then input sample total
        to_undersample = cat_df.value_counts('category')[cat_df.value_counts('category') > sample_total].index
    
        test_images= []
        #for each of those categories calculate how many to drop, sample that number, and drop them
        for category_name in to_undersample:
        
            category = cat_df.loc[cat_df['category'] == category_name]
            num_drop = len(category) - sample_total
        
            #sample randomly or by slice
            if bool(random):
                drop_subset = cat_df.loc[cat_df['category'] == category_name].sample(num_drop)
            else:
                drop_subset = cat_df.loc[cat_df['category'] == category_name][:num_drop]
            
            #drop the subset
            cat_df.drop(drop_subset.index, inplace = True)
        
            #joining random images from subset to create a test image list
            test_images.append(drop_subset.iloc[0])
            
    else:
        #save categories containing less then input sample total
        to_undersample = cat_df.value_counts('sub_category')[cat_df.value_counts('sub_category') > sample_total].index
    
        test_images= []
        #for each of those categories calculate how many to drop, sample that number, and drop them
        for category_name in to_undersample:
        
            category = cat_df.loc[cat_df['sub_category'] == category_name]
            num_drop = len(category) - sample_total
        
            #sample randomly or by slice
            if bool(random):
                drop_subset = cat_df.loc[cat_df['sub_category'] == category_name].sample(num_drop)
            else:
                drop_subset = cat_df.loc[cat_df['sub_category'] == category_name][:num_drop]
            
            #drop the subset
            cat_df.drop(drop_subset.index, inplace = True)
        
            #joining random images from subset to create a test image list
            test_images.append(drop_subset.iloc[0])
    
    return cat_df, test_images

 
#visualizing image categories
    
def explore_tfds_cats(cat_key, category, preview = 4, sub_category = False):  
    """Visualizes photos from specific category or sub category of dataframe with images and labels
    
    INPUT
    ----
    
    OUTPUT
    -----
    
    """
    import matplotlib.pyplot as plt
     
    if bool(sub_category):
        index = cat_key.loc[cat_key['sub_category'] == sub_category].index
        labels = cat_key.loc[cat_key['sub_category'] == sub_category].label
    else:
        index = cat_key.loc[cat_key['category'] == category].index
        labels = cat_key.loc[cat_key['category'] == category].label

    #for up to inputted range show image previews
    for i in index[:preview]:
        plt.imshow(cat_key.image[i])
        plt.figure(i+1) 