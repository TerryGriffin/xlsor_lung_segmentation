"""
Author Xinzi Sun University of Massachusetts Lowell
"""
import imageio
import os
import matplotlib.pyplot as plt
from skimage import io, transform

original_images = '/data0/dataset/xray/xray_images'
segmentation = '/data0/dataset/xray/segmentation_outputs'
original_images_with_segmentation = '/data0/dataset/xray/original_images_with_segmentation'

i = 1
for image in os.listdir(original_images):
    try:
        img = io.imread(os.path.join(original_images, image))
        mask = io.imread(os.path.join(segmentation, image))
        mask = transform.resize(mask, img.shape[0:2])
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.savefig(os.path.join(original_images_with_segmentation, image))
        plt.clf()
        print(i)
    except:
        print(image)
    i += 1
