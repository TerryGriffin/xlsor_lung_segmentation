from argparse import ArgumentParser
import os
import sys
import pathlib
import numpy as np
import logging
import cv2
import torch.nn as nn
import torch.distributed
import torch
from networks.xlsor import XLSor
from PIL import Image as PILImage
from skimage import io, transform


logger = logging.getLogger("segment_cxr")
logger.setLevel(logging.INFO)


def init_distributed():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    backend = "gloo"
    rank = 0
    size = 1
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)


def load_image(filename):
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    try:
        image = cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_CUBIC)
    except:
        return None

    size = image.shape
    image = np.asarray(image, np.float32)
    image -= IMG_MEAN

    crop_h, crop_w = (512, 512)
    img_h, img_w, _ = image.shape
    pad_h = max(crop_h - img_h, 0)
    pad_w = max(crop_w - img_w, 0)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0)
        )

    image = image.transpose((2, 0, 1))
    return original_image, image, np.array(size)


def segment_cxr(input_filename, output_filename, mask_filename, model_params, gpu):
    init_distributed()

    original_image, image, image_size = load_image(input_filename)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)

    model = XLSor(1)
    saved_state_dict = torch.load(model_params, map_location=torch.device("cpu"))
    model.load_state_dict(saved_state_dict)
    model.eval()

    if gpu is not None and torch.cuda.is_available():
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    else:
        use_gpu = False

    if use_gpu:
        model.cuda()
        image = image.cuda()

    with torch.no_grad():
        mask = model(image, 2)
        if isinstance(mask, list):
            mask = mask[0]

        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        mask = interp(mask).cpu().data[0].numpy()
        mask = mask.transpose(1, 2, 0)

    if mask_filename:
        output_im = PILImage.fromarray((np.clip(mask[:,:,0],0,1)* 255).astype(np.uint8))
        output_im.save(mask_filename)

    mask = transform.resize(mask, original_image.shape[0:2])
    mask = (mask >= 0.5).astype(np.uint8)
    masked = cv2.add(original_image, np.zeros(np.shape(original_image), dtype=np.uint8), mask=mask)
    cv2.imwrite(output_filename, masked)
    print("here")


if __name__ == "__main__":
    script_path = pathlib.Path(__file__).parent.absolute()
    default_model_params = os.path.join(script_path, "models/XLSor.pth")
    parser = ArgumentParser(description="Segment lung area")
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="output file")
    parser.add_argument(
        "--mask-output", type=str, default=None, help="mask output file"
    )
    parser.add_argument(
        "--model", type=str, default=default_model_params, help="model params"
    )
    parser.add_argument("--gpu", type=str, default=None, help="gpu to use")
    args = parser.parse_args()
    try:
        segment_cxr(args.input, args.output, args.mask_output, args.model, args.gpu)
    except Exception as ex:
        logger.exception(f"Exception from segment_cxr: {ex}")
        sys.exit(1)
    sys.exit(0)
