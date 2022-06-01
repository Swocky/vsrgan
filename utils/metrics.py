import argparse
import math
import numpy as np
import os
from PIL import Image

import matlab
import matlab.engine
from skimage.metrics import structural_similarity


def compute_psnr(clean_img, img, data_range):
    """compute the psnr
    """
    clean_img = np.array(clean_img).astype(np.float32)
    img = np.array(img).astype(np.float32)
    mse = np.mean((clean_img - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = data_range
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_ssim(clean_img, img, data_range=255, multichannel=True):
    return structural_similarity(clean_img, img, data_range=data_range, multichannel=multichannel)


if __name__ == "__main__":
    eng = matlab.engine.start_matlab()
    out_root = 'results/'
    gt_root = 'data/LR/val/'
    psnr_sum = 0
    ssim_sum = 0
    niqe_sum = 0
    for i in range(20):
        psnr = 0
        ssim = 0
        niqe = 0
        for j in range(29):
            img_1 = np.array(Image.open(os.path.join(out_root, '%03d'%i, '%03d.png'%(j + 1))))
            img_2 = np.array(Image.open(os.path.join(gt_root, 'val_%03d'%i, 'truth', '%03d.png'%(j + 1))))
            h = min(img_1.shape[0], img_2.shape[0])
            w = min(img_1.shape[1], img_2.shape[1])
            psnr += compute_psnr(img_1[:h,:w,:], img_2[:h,:w,:], data_range=255)
            ssim += compute_ssim(img_1[:h,:w,:], img_2[:h,:w,:], data_range=255, multichannel=True)
            niqe += eng.cal_niqe(os.path.join(out_root, '%03d'%i, '%03d.png'%(j + 1)))
        psnr_sum += psnr / 29
        ssim_sum += ssim / 29
        niqe_sum += niqe / 29
        print('PSNR:', psnr / 29, 'SSIM', ssim / 29, 'NIQE', niqe / 29)
    print('Mean PSNR:', psnr_sum / 20, 'Mean SSIM', ssim_sum / 20, 'Mean NIQE', niqe_sum / 20)
