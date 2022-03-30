import numpy as np
from tqdm import tqdm
import argparse

from feature_extractors.ResNetFeatureExtractor import ResNetFeatureExtractor
from datasets.BaseDataset import BaseDataset

def save_features_test(data_dir):
    
    dataset = BaseDataset(data_dir)
    feature_extractor = ResNetFeatureExtractor()
    features = []
    for sample_path in tqdm(dataset.sample_paths()):
        features.append(feature_extractor.get_features(sample_path))
    
    dataset.save_features(features)


# Parse arguments
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group('required')
    req_grp.add_argument('--data_dir', default=None, help='directory of data to be clustered.')
    args = parser.parse_args()

    save_features_test(args.data_dir)