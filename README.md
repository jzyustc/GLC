<div align="center">

# Generative Latent Coding for Ultra-Low Bitrate Image Compression

</div>

[![CVPR 2024](https://img.shields.io/badge/Paper-CVPR%202024-blue?logo=readthedocs&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2024/papers/Jia_Generative_Latent_Coding_for_Ultra-Low_Bitrate_Image_Compression_CVPR_2024_paper.pdf)


Official Implementation of GLC, Generative Latent Coding for Ultra-Low Bitrate Image Compression.

## Introduction

Most existing image compression approaches perform transform coding in the pixel space to reduce its spatial redundancy. However, they encounter difficulties in achieving both high-realism and high-fidelity at low bitrate, as the pixel-space distortion may not align with human perception. To address this issue, we introduce a Generative Latent Coding (GLC) architecture, which performs transform coding in the latent space of a generative vector-quantized variational auto-encoder (VQ-VAE), instead of in the pixel space. The generative latent space is characterized by greater sparsity, richer semantic and better alignment with human perception, rendering it advantageous for achieving high-realism and high-fidelity compression. Additionally, we introduce a categorical hyper module to reduce the bit cost of hyper-information, and a code-prediction-based supervision to enhance the semantic consistency. Experiments demonstrate that our GLC maintains high visual quality with less than 0.04 bpp on natural images and less than 0.01 bpp on facial images. On the CLIC2020 test set, we achieve the same FID as MS-ILLM with 45% fewer bits. Furthermore, the powerful generative latent space enables various applications built on our GLC pipeline, such as image restoration and style transfer.

<img src="assets/pipeline.png" width="750">


## Compression Performance

Visual comparison :

<img src="assets/visual.png" width="750">

RD-Curves on CLIC2020 : 

<img src="assets/rd.png" width="750">

Please refer to the paper for more details.


## :hammer: Test Pretrained Models

Prepare the conda environment:

```bash
conda create -n glc python=3.12
conda activate glc
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Download the pretrained weights in the release page, config the paths correctly and run `test.py`

```bash
python test.py
```


## :page_facing_up: Citation
If you find this work useful for your research, please cite:
```
@inproceedings{jia2024generative,
  title={Generative latent coding for ultra-low bitrate image compression},
  author={Jia, Zhaoyang and Li, Jiahao and Li, Bin and Li, Houqiang and Lu, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26088--26098},
  year={2024}
}
```


## Acknowledgement

The main implementation of GLC is based on [DCVC](https://github.com/InterDigitalInc/CompressAI), the code prediction part is based on [CodeFormer](https://github.com/sczhou/CodeFormer) and the metric evaluation part is based on [NeuralCompression](https://github.com/facebookresearch/NeuralCompression).
