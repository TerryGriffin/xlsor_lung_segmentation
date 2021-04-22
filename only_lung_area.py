"""
Author Xinzi Sun University of Massachusetts Lowell
"""
import imageio
import os
import matplotlib.pyplot as plt
from skimage import io, transform
import cv2
import numpy as np

# original_images = '/data2/xinzi/CADLab/Lung_Segmentation_XLSor/data/xray/xray_images'
# segmentation = '/data2/xinzi/CADLab/Lung_Segmentation_XLSor/outputs'
# original_images_with_segmentation = '/data2/xinzi/CADLab/Lung_Segmentation_XLSor/data/xray/xray_images_only_lung_area'

original_images = '/data2/xinzi/SOLO/data/pleural_effusion_and_others'
segmentation = '/data2/xinzi/CADLab/Lung_Segmentation_XLSor/data/CheXpert_mask/pleural_effusion_and_others'
original_images_with_segmentation = '/data2/xinzi/CADLab/Lung_Segmentation_XLSor/data/CheXpert/SUBSETS-small/Pleural_Effusion_and_Others/images'

i = 1
for image in os.listdir(original_images):
    # print(os.path.join(segmentation, image[:-4] + '_xlsor.jpg'))
    try:
        img = cv2.imread(os.path.join(original_images, image)).astype(np.uint8)
    except:
        print(os.path.join(original_images, image))
    mask = cv2.imread(os.path.join(segmentation, image[:-4] + '_xlsor.png'), cv2.IMREAD_GRAYSCALE)
    # mask = cv2.imread(os.path.join(segmentation, image), cv2.IMREAD_GRAYSCALE)
    mask = transform.resize(mask, img.shape[0:2])
    mask = (mask >= 0.5).astype(np.uint8)
    masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    cv2.imwrite(os.path.join(original_images_with_segmentation, image), masked)
    
    i += 1
