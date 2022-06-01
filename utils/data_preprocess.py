import glob
import os
from PIL import Image
from tqdm import *


if __name__ == "__main__":
    paths = glob.glob('data/HR/**/*.jpg', recursive=True)
    paths.sort()
    save_dir = 'data/LR/'
    scale = 8
    for path in tqdm(paths):
        hr = Image.open(path).convert('RGB')
        w, h = hr.size
        lr = hr.resize((w//scale, h//scale))
        save_path = path.replace('HR', 'LR')
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
          os.makedirs(save_dir)
        lr.save(save_path)
