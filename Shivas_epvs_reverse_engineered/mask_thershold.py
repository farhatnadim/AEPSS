''' this file takes as input the masks that have probabilities of being a vessel and outputs a mask with only the voxels that have a probability of being a vessel above a certain threshold'''
''' this file reads the config.json file to get the mask_threshold'''
''' this file reads config.json to get the probabilistic_masks_dir and the thersholded_masks_dir'''
import json
import os
import SimpleITK as sitk

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
        
        return config
    
def threshold_masks(probabilistic_masks_dir, thresholded_masks_dir, mask_threshold):
    ''' this function takes as input the probabilistic_masks_dir, the thresholded_masks_dir and the mask_threshold'''
    ''' this function thresholds the masks and saves them in the thresholded_masks_dir'''
    for file in os.listdir(probabilistic_masks_dir):
        if file.endswith('.nii'):
            mask = sitk.ReadImage(os.path.join(probabilistic_masks_dir, file))
            mask_array = sitk.GetArrayFromImage(mask)
            mask_array[mask_array < mask_threshold] = 0
            mask_array[mask_array >= mask_threshold] = 1
            mask = sitk.GetImageFromArray(mask_array)
            sitk.WriteImage(mask, os.path.join(thresholded_masks_dir, file))

def main():
    config = load_config('config.json')
    probabilistic_masks_dir = config['probabilistic_masks_dir']
    thresholded_masks_dir = config['thresholded_masks_dir']
    mask_threshold = config['mask_threshold']
    threshold_masks(probabilistic_masks_dir, thresholded_masks_dir, mask_threshold)

if __name__ == '__main__':
    main()
    
