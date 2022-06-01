import argparse
import glob
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import *

from utils.warp_utils import flow_warp
from networks.generator import Generator
from networks.raft.raft import RAFT
from networks.raft.utils.utils import InputPadder



def demo(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    gen_path = args.gen_model
    raft_path = args.raft_model

    device = torch.device('cuda')
    test_img_folder = os.path.join(args.test_dir, '*.png')

    generator = Generator(3, 3, 64, 23, gc=32).to(device)
    generator.load_state_dict(torch.load(gen_path), strict=True)
    generator.eval()

    raft = RAFT()
    raft = torch.nn.DataParallel(raft).to(device)
    raft.load_state_dict(torch.load(raft_path))
    raft.eval()

    hrs = None
    lrs = None
    paths = glob.glob(test_img_folder)
    paths.sort()
    for path in tqdm(paths):
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        lr = img.unsqueeze(0)
        lr = lr.to(device)
        lrs = lr if lrs == None else torch.cat((lrs, lr), axis=0)
        with torch.no_grad():
            hr = generator(lr) # b, c, h, w
        hrs = hr if hrs == None else torch.cat((hrs, hr), axis=0)

    n = hrs.shape[0]
    ws = args.wsize
    assert n >= ws
    half_win = (ws - 1) // 2
    for offset in tqdm(range(n - ws)):
        with torch.no_grad(): 
            mid = offset + (ws - 1) // 2
            lr_l = lrs[offset,:,:,:].unsqueeze(0)
            lr_mid = lrs[mid,:,:,:].unsqueeze(0)
            lr_r = lrs[offset+ws-1,:,:,:].unsqueeze(0)
            padder = InputPadder(lr_l.shape)
            lr_l_pad, lr_mid_pad = padder.pad(lr_l, lr_mid)
            lr_r_pad, _ = padder.pad(lr_r, lr_r)
            padder = InputPadder(hrs.shape)
            hrs_pad, _ = padder.pad(hrs, hrs)
            _, of_l = raft(lr_mid_pad, lr_l_pad, iters=20, test_mode=True)
            _, of_r = raft(lr_mid_pad, lr_r_pad, iters=20, test_mode=True)
            of_l = F.interpolate(of_l, scale_factor=4)[:,:,:hrs_pad.shape[2],:hrs_pad.shape[3]] * 4 / half_win
            of_r = F.interpolate(of_r, scale_factor=4)[:,:,:hrs_pad.shape[2],:hrs_pad.shape[3]] * 4 / half_win
            frame_sum = hrs_pad[mid,:,:,:].unsqueeze(0)
            for i in range(ws):
                index = offset + i
                cur_x = hrs_pad[index,:,:,:].unsqueeze(0)
                if index != mid:
                    of = (mid - index) * of_l if index < mid else (index - mid) * of_r
                    cur_x = flow_warp(cur_x, of)
                    frame_sum = frame_sum + cur_x
            output = frame_sum / ws
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(os.path.join(args.out_dir, '%03d.png'%(offset+half_win)), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Super-Resolution GAN")
    parser.add_argument('--wsize', type=int, default=3, help='window size for the optical flow (default: 3)')
    parser.add_argument('--test_dir', nargs='?', type=str, default='data/LR/val/val_000/blur4/',
                        help='the path of the video frames (default: data/LR/val/val_000/blur4/)')
    parser.add_argument('--out_dir', nargs='?', type=str, default='results',
                        help='the path of the video frames (default: results)')
    parser.add_argument('--gen_model', nargs='?', type=str, default='weights/generator.pth',
                        help='path to saved generator model (default: weights/generator.pth)')
    parser.add_argument('--raft_model', nargs='?', type=str, default='weights/raft.pth',
                        help='path to saved raft model (default: weights/raft.pth)')
    args = parser.parse_args()
    demo(args)
