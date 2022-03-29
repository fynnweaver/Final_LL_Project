# Final_LL_Project
Combining an image recommender engine with duplicate detection neural net for suggestions only containing original content.

To run image recommender follow these steps:
1. Clone git repo
2. Run `DataCollection.ipynb` (takes roughly 4 hours total) OR download and unzip a reference image directory of 6K images [here](). The folder containing images should be in the directory above the repository and have the following structure:

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

The original `distribution_clustering` repo can be found [here]()
