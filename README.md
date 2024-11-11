## LIEDNet: A Lightweight Network for Low-light Enhancement and Deblurring

AIR, Technical University of Munich

>**Abstract:** Images captured at nighttime often face challenges such as low light and blur, primarily caused by dim environments and the frequent use of long exposure. Existing methods either handle the two types of degradations independently or rely on carefully designed priors generated by complex mechanisms, resulting in poor generalization ability and high model complexity. To address these challenges, we propose an end-to-end framework named LIEDNet to efficiently and effectively restore high-quality images on both real-world and synthetic data. Specifically, the introduced LIEDNet consists of three essential components: the Visual State Space Module (VSSM), the Local Feature Module (LFM), and the Dual Gated-Dconv Feedforward Network (DGDFFN). The integration of VSSM and LFM enables the model to capture both global and local features while maintaining low computational overhead. Additionally, the DGDFFN improves image fidelity by extracting multi-scale structural information. Extensive experiments on real-world and synthetic datasets demonstrate the superior performance of LIEDNet in restoring low-light, blurry images. 

---

### Results
<details close>
<summary><b>Performance vs. efficiency of deblurring models:</b></summary>

![results1](/figures/performance_complexity_res.png)
</details>

<details close>
<summary><b>Performance on LOL-Blur:</b></summary>

![results_lolblur](/figures/LOL_Blur_res.png)

</details>

<details close>
<summary><b>Performance on Real-LOL-Blur:</b></summary>

![results_reallolblur](/figures/Real_LOL_Blur_res.png)

</details>

<details close>
<summary><b>Performance on LOL-v1, LOL-v2-synthetic, and FiveK:</b></summary>

![results_llie](/figures/LLIE_res.png)

</details>

### LOL-Blur Dataset
The datasets are provided by [LEDNet](https://github.com/sczhou/LEDNet).
| Dataset | Link |
| :----- | :--: |
| LOL-Blur | [Google Drive](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX?usp=sharing) / [BaiduPan (key: dz6u)](https://pan.baidu.com/s/1CPphxCKQJa_iJAGD6YACuA) |
| Real-LOL-Blur| [Google Drive](https://drive.google.com/drive/folders/1fXUA5SzXj46ISw9aUjSors1u6M9VlKAn?usp=sharing) / [BaiduPan (key: fh32)](https://pan.baidu.com/s/1sP87VGiof_NixZsA8dhalA) |

### Low-light Enhancement Datasets
These datasets are provided by [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer)
|Dataset | Link |
| :----- | :--: |
|LOL-v1 | [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing) / [BaiduPan (key: cyh2)](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) |
|LOL-v2 | [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing) / [BaiduPan (key: cyh2)](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) | 
|FiveK  | [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing) / [BaiduPan (key: cyh2)](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) | 
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
| LOL-Blur | [Google Drive](https://drive.google.com/file/d/159Cyv2jQAu6j6-jXA_N4Y3fEPUZz6587/view?usp=sharing) |
| LOL-Blur (Large) | [Google Drive](https://drive.google.com/file/d/1vp0bVeX2lZSGq9Ohc2QbVOd5vNI7Vb3U/view?usp=sharing) |
| LOL-v1 | [Google Drive](https://drive.google.com/file/d/1CWNmUFWZLFLvj0M1nEjgZ9CeIVJZ44Bb/view?usp=sharing) |
| LOL-v2-synthetic | [Google Drive](https://drive.google.com/file/d/1ZOb6pG4iwF0EswBIBKC7680ltfDPYjbM/view?usp=sharing) |
| FiveK | [Google Drive](https://drive.google.com/file/d/15qxh-_iGKgLL8SmYpyGuJH62SeU8LpIJ/view?usp=sharing) |
  
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
