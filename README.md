## LIEDNet: A Lightweight Network for Low-light Enhancement and Deblurring

AIR, Technical University of Munich

---

### LOL-Blur Dataset
The datasets are provided by [LEDNet](https://github.com/sczhou/LEDNet. The datasets are hosted on both Google Drive and BaiduPan)
| Dataset | Link |
| :----- | :--: |
| LOL-Blur | [Google Drive](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX?usp=sharing) / [BaiduPan (key: dz6u)](https://pan.baidu.com/s/1CPphxCKQJa_iJAGD6YACuA) | 12,000 | A total of 170 videos for training and 30 videos for testing, each of which has 60 frames, amounting to 12,000 paired data. (Note that the first and last 30 frames of each video are NOT consecutive, and their darknesses are simulated differently as well.)|
| Real-LOL-Blur| [Google Drive](https://drive.google.com/drive/folders/1fXUA5SzXj46ISw9aUjSors1u6M9VlKAn?usp=sharing) / [BaiduPan (key: fh32)](https://pan.baidu.com/s/1sP87VGiof_NixZsA8dhalA) | 1354 | 482 real-world night blurry images (from [RealBlur-J Dataset](http://cg.postech.ac.kr/research/realblur/)) + 872 real-world night blurry images acquired by Sony RX10 IV camera.|

### Dependencies and Installation

- Pytorch >= 1.13.1
- CUDA >= 11.6
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/MingyuLiu1/LIEDNet.git
cd LIEDNet

# create new anaconda env
conda create -n liednet python=3.10 -y
conda activate liednet

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop

cd basicsr/kernels/selective_scan
python install .
```

### Train the Model
- Specify the path to training and evaluation data in the corresponding option file.

Training LIEDNet:
```
# LOL-Blur
python train.py --opt options/train_LOLBLur.yml
```

```
# LOL-v1
python train_llie.py --opt options/train_LOLv1.yml 
```

```
# LOL-v2-syn
python train_llie.py --opt options/train_LOLv2_synthetic.yaml
```

```
# FiveK
python train_llie.py --opt options/train_fivek.yaml
```

### Quick Inference
- Download the LIEDNet pretrained model from:
  
| Pre-trained checkpoint | Link |
| :-: | :-: |
| LOL-Blur | [Google Drive](https://drive.google.com/file/d/179NVjpyg0PHMI4lMQaXYGQSNbQqi7U6U/view?usp=sharing) |
| LOL-Blur (Large) | [Google Drive]() |
| LOL-v1 | [Google Drive](https://drive.google.com/file/d/1M_7mugY-V0DjY6EgA-JXilH0QUwNyYyh/view?usp=sharing) |
| LOL-v2-synthetic | [Google Drive](https://drive.google.com/file/d/1yfvGVJvCjGYzIzCnNxzH38Ue3zGi9X4P/view?usp=sharing) |
| FiveK | [Google Drive](https://drive.google.com/file/d/1iNWlKO7Utav2oHCtIkmUYI4gZMVtoYfc/view?usp=sharing) |
  
Inference LIEDNet (save images):
```
# test LIEDNet on LOL-Blur
python inference_lol_blur.py --test_path $INPUT PATH$ --result_path $SAVE PATH$ --ckpt $CHECKPOINT PATH$

# test LIEDNet on LLIE datasets (LOL-V1, LOL-V2-synth, FiveK)
python inference_llie.py --test_path $INPUT PATH$ --result_path $SAVE PATH$ --ckpt $CHECKPOINT PATH$
```
The results will be saved in $SAVE PATH$.

### Evaluation

```
# Evaluation on synthetic datsets (LOL-Blur, LOL-V1, LOL-V2-synthetic, FiveK). Set evaluation metrics of 'psnr', 'ssim', and 'lpips (vgg)'
python calculate_pair --result_path $RESULT PATH$ --gt_path $GROUND TRUTH PATH$ --metrics psnr ssim lpips
```

```
# Evaluation on the real-world Real-LOL-Blur dataset
python calculate_unpair.py --result_path $RESULT PATH$ --gt_path $GROUND TRUTH PATH$ -m musiq nrqm niqe
```


### License

This project is licensed under <a rel="license" href="https://github.com/MingyuLiu1/LIEDNet/blob/main/LICENSE">Apache License 2.0</a>. Redistribution and use for non-commercial purposes should follow this license.

### Acknowledgement

This project is built on [LIEDNet](https://github.com/sczhou/LEDNet), [LLFormer](https://github.com/TaoWangzj/LLFormer), [VMamba](https://github.com/MzeroMiko/VMamba), and [VQCNIR](https://github.com/AlexZou14/VQCNIR). 

### Contact
If you have any questions, please feel free to reach me out at `mingyu.liu@tum.de`.
