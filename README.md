# Final_LL_Project
Combining an image recommender engine with duplicate detection for suggestions of visually similar images only containing original content.

To run image recommender follow these steps:
1. Clone git repo
2. a) Run `DataCollection.ipynb`. This grabs images from several tensorflow dataset and saves a subsample to `image_dir` in the folder above the repo. Depending on how many images are included and processing power this can take from several hours to an entire day for encoding and clustering.  
   b) OR download and unzip a reference image directory of 10K images [here](https://drive.google.com/file/d/1rpuAit9J1vBWU0gvCXfOQpXE01O6X1wL/view?usp=sharing). 
Either way, the folder containing images should be in the directory above the repository and have the following structure:

```
- <data_dir>
    - samples
          - features.npy
          - centers.npy
          - pics
             - <first file>
             - ...
```
3. From the terminal within the project repository run `python recommend_images.py return_recommendation <path to input image directory> <path to reference images directory>`
4. Recommendations will be saved as a jpeg in the input image directory

To explore the pieces of the `return_recommendation` functions use the `BuildingRecommender.ipynb` notebook. 

The original `distribution_clustering` repo is thanks to [Eric Elmoznino](https://github.com/EricElmoznino) and can be found [here](https://github.com/EricElmoznino/distribution_clustering)
