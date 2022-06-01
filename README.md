Video Super-Resolution GAN
=================
A generative method for video super-resolution which using the motion filter to make the results more stable.

Pipeline
-----------------
![pipeline](imgs/pipeline.png)

Dependencies
-----------------
- numpy==1.22.4
- opencv_python==4.5.4.60
- Pillow==9.1.1
- scipy==1.8.1
- torch==1.8.0
- torchvision==0.9.0
- tqdm==4.64.0


Test Model
-----------------
```bash
python demo.py --test_dir your_path --out_dir your_path
```
You can also use a bash script to run the demo as exampled in 'run.sh'.

Pretrained Models Download
-----------------
The pretrained model can be download from google drive.
- [GAN](https://drive.google.com/drive/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY)
- [RAFT](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

Remeber to use the command line arguments to locate the models.
```bash
python demo.py --gen_model your_path --raft_model your_path
```

Dataset Download
-----------------
Following ESRGAN, we use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flick2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) to train the model.

The evaluation dataset can be download from baidu drive, and we use [val.zip](https://pan.baidu.com/s/1TNtkn_dHHQf_3_JABKWgZg#list/path=%2F) (password: pr7p) in our experiments. Don't forget to set --test_dir for demo.py.


Quantitative Results
-----------------
We use the 20 videos in val.zip for quantitative evaluation. Each value is averaged from all the frames (with three channels) of a video. 
| video | PSNR | SSIM | NIQE |
|------|------|------|------|
| val_000 | 18.17 | 0.5253 | 2.9197 |
| val_001 | 26.11 | 0.7897 | 4.1745 |
| val_002 | 20.99 | 0.5573 | 2.2559 |
| val_003 | 21.29 | 0.4760 | 2.6479 |
| val_004 | 23.15 | 0.6637 | 2.9329 |
| val_005 | 22.77 | 0.6415 | 2.5267 |
| val_006 | 27.03 | 0.6821 | 3.3715 |
| val_007 | 24.24 | 0.6394 | 3.2039 |
| val_008 | 23.80 | 0.6187 | 2.7857 |
| val_009 | 24.63 | 0.7498 | 3.3803 |
| val_010 | 22.70 | 0.5712 | 2.9139 |
| val_011 | 23.41 | 0.6392 | 3.3336 |
| val_012 | 22.60 | 0.6127 | 2.8355 |
| val_013 | 22.86 | 0.6389 | 3.0100 |
| val_014 | 22.82 | 0.6369 | 2.9201 |
| val_015 | 26.08 | 0.7792 | 3.9357 |
| val_016 | 21.81 | 0.5416 | 2.3911 |
| val_017 | 23.57 | 0.6128 | 3.6706 |
| val_018 | 22.79 | 0.5165 | 3.6781 |
| val_019 | 21.18 | 0.5046 | 3.4247 |
| average | 23.10 | 0.6199 | 3.1156 |

Visual Results
-----------------
- Low Resolution

![002_lr](imgs/002_lr.gif)

- High Resolution

![002_hr](imgs/002_hr.gif)

- Low Resolution

![003_lr](imgs/003_lr.gif)

- High Resolution

![003_hr](imgs/003_hr.gif)

- Low Resolution

![006_lr](imgs/006_lr.gif)

- High Resolution

![006_hr](imgs/006_hr.gif)


Acknowledge
-----------------
- [RAFT](https://github.com/princeton-vl/RAFT)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [ESRGAN](https://github.com/xinntao/ESRGAN)
- [MSHPFNL](https://github.com/psychopa4/MSHPFNL)