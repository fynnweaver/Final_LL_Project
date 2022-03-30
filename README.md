# Final_LL_Project
Combining an image recommender engine with duplicate detection for suggestions of visually similar images only containing original content.

To run image recommender follow these steps:
1. Clone git repo
2. a) Download and unzip a reference image directory of 10K images [here](https://drive.google.com/file/d/1rpuAit9J1vBWU0gvCXfOQpXE01O6X1wL/view?usp=sharing). The folder is already set up with embeddings and clusters.  
   b) OR Run `DataCollection.ipynb`. This grabs images from several tensorflow dataset and saves a subsample to `image_dir` in the folder above the repo. Depending on how many images are included and processing power this can take from several hours to an entire day for encoding and clustering.  
   For either method, the folder containing images should have the following structure:

```
- <data_dir>
    - features.npy
    - centers.npy
    - samples
             - <first file>
             - ...
```
3. Create a new folder and save the input image to recommend from into an empty subfolder named `samples`
4. From the terminal within the project repository run `python recommend_images.py return_recommendation <path to input image directory> <path to reference images directory>`
5. Recommendations will be saved as a jpeg in the input image directory

To explore the pieces of the `return_recommendation` functions use the `BuildingRecommender.ipynb` notebook. 

The original `distribution_clustering` repo is thanks to [Eric Elmoznino](https://github.com/EricElmoznino) and can be found [here](https://github.com/EricElmoznino/distribution_clustering)
