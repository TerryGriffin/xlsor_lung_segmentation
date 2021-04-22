'''
Youbao Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
April 2019

For testing, you need to modify some arguments according to your own setting and run the command "python test.py".

Modified by Xinzi Sun University of Massachusetts Lowell
Modified by Terry Griffin University of Massachusetts Lowell
'''



import argparse
import numpy as np

import torch
from torch.utils import data
from networks.xlsor import XLSor
from dataset.datasets import XRAYDataTestSet
import os
from PIL import Image as PILImage

import torch.nn as nn
import torch.distributed

USE_CUDA = True

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 1

# DATA_DIRECTORY = './data/xray/xray_images'
# DATA_LIST_PATH = 'png'
# RESTORE_FROM = './models/XLSor.pth'

DATA_DIRECTORY = './test_data'
DATA_LIST_PATH = 'png'
RESTORE_FROM = './models/XLSor.pth'

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    parser.add_argument("--output", type=str, default="outputs",
                        help="output directory")
    parser.add_argument("--recursive", action="store_true",
                        help="recurse into subdirectories")
    return parser.parse_args()


def main():
    args = get_arguments()

    out_dir = args.output
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    backend = 'gloo'
    rank=0
    size=1
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)

    if USE_CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model = XLSor(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from, map_location=torch.device("cpu"))
    model.load_state_dict(saved_state_dict)

    model.eval()
    if USE_CUDA:
        model.cuda()

    if args.recursive:
        process_tree(model, args.data_dir, out_dir, args)
    else:
        process_dir(model, args.data_dir, out_dir, args)

def process_tree(model, root_dir, out_dir, args):
    process_dir(model, root_dir, out_dir, args)
    for filename in os.listdir(root_dir):
        full_path = os.path.join(root_dir, filename)
        if os.path.isdir(full_path):
            process_tree(model, full_path, os.path.join(out_dir, filename), args)

def process_dir(model, image_dir, out_dir, args):
    testloader = data.DataLoader(XRAYDataTestSet(image_dir, args.data_list, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)

    if len(testloader) == 0:
        return

    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processed'%(index))
        image, size, name = batch
        if USE_CUDA:
            image = image.cuda()
        with torch.no_grad():
            prediction = model(image, args.recurrence)
            if isinstance(prediction, list):
                prediction = prediction[0]
            prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
        output_im = PILImage.fromarray((np.clip(prediction[:,:,0],0,1)* 255).astype(np.uint8))

        output_name = os.path.join(out_dir, os.path.splitext(os.path.basename(name[0]))[0] + "_mask.png")
        output_im.save(output_name, 'png')



if __name__ == '__main__':
    main()
