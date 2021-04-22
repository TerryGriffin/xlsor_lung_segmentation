"""
Author Xinzi Sun University of Massachusetts Lowell
"""
import os

segmentation = '/data2/xinzi/CADLab/Lung_Segmentation_XLSor/outputs'
for image in os.listdir(segmentation):
	old = os.path.join(segmentation, image)
	new = image.split('_')[0]
	new = new + '.png'
	new = os.path.join(segmentation, new)
	os.rename(old, new)
