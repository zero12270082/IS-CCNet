# Illumination Separation and Adaptive Color Compensation: Enhancing Thangka Image Exposure Correction
<p align="center">
    <p align="center">
        <a >WU Ji-yuan</a>
        ·
        <a >ZHANG Xiao-juan</a>

<div align="center">


[![DOI](https://zenodo.org/badge/DOI/DOI.svg)](https://doi.org/10.5281/zenodo.15582883)
[![LCDP](https://img.shields.io/badge/Dataset-LCDP-%23cda6c3)](https://github.com/onpix/LCDPNet/tree/main)
[![MSEC](https://img.shields.io/badge/Dataset-MSEC-%23cda6c3)](https://github.com/mahmoudnafifi/Exposure_Correction)


</div>


> **🔬 Official implementation of the manuscript submitted to *The Visual Computer***  
> **"Illumination Separation and Adaptive Color Compensation: Enhancing Thangka Image Exposure Correction"**  
> **Submission ID: ` `**  
> **DOI: ` https://doi.org/10.5281/zenodo.15582883 `**  
> **📌 If you use this code in your research, please cite our work!**

This is the official implementation of the paper *"Illumination Separation and Adaptive Color Compensation: Enhancing Thangka Image Exposure Correction"*. The code is implemented in PyTorch.


**Abstract**: Thangka, a UNESCO intangible cultural heritage, holds immense artistic and historical value. However, uneven lighting during photography often leads to color distortion and loss of details in Thangka images. Existing enhancement methods struggle to restore image details and color fidelity under complex lighting
conditions. This paper introduces a novel exposure correction framework for Thangka images, leveraging illumination separation and adaptive color compensation. We curated a specialized Thangka image dataset for model training and validation. The framework includes an Illumination Separation Analysis Module (ISAM) and an Adaptive Convolutional Color Compensation Module (ACCM) to address exposure imbalance and chromatic deviations, respectively. Furthermore, a Wave-Attention Feature Enhancement Module (WAEM) is proposed to extract multifrequency features, enhancing both textural and chromatic details. Experimental results on public benchmarks and our Thangka dataset demonstrate superior performance, with our method achieving average improvements of 12.3% in PSNR and 4.9% in SSIM over existing approaches. This work not only advances the field of image exposure correction but also contributes to the digital preservation of cultural heritage. Comprehensive experiments show that our method outperforms existing approaches.

##  News
- [2025/06/03] Update Google Drive link for the paper and README.Release training and testing code.


## Installation
To get started, clone this project, create a conda virtual environment using Python 3.9 (or higher versions may do as well), and install the requirements:
```
git clone https://github.com/zero12270082/IS-CCNet.git
cd  ISCC
conda create -n ISCC python=3.9
conda activate ISCC
# Change the following line to match your environment
# Reference: https://pytorch.org/get-started/previous-versions/#v1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```


##  Running the Code

### Evaluation

To evaluate the trained model, you'll need to do the following steps:
- Get the [pretrained model](https://drive.google.com/drive/folders/1qIxYuPt1OtYZ0yGMLcOLzwbDR_V0ZU3D) (or you can use your own trained weights) and put them in the `pretrained/` folder.
- Modify the path to the test dataset in `src/config/ds/test.yaml` (if you don't need ground truth images for testing, just leave the `GT` value as `none`).
- Run the following command:
    ```
    python src/test.py checkpoint_path=path/to/checkpoint/filename.ckpt
    ```
- The results will be saved in the `test_result/` folder under `path/to/checkpoint/`.

### Training

To train your own model from scratch, you'll need to do the following steps:
- Prepare the training dataset. You can use the [LCDP dataset](https://github.com/onpix/LCDPNet/tree/main) or [MSEC dataset](https://github.com/mahmoudnafifi/Exposure_Correction) (or you can use your own paired data)，It is worth noting that all the thangka datasets mentioned in the article are collected from field research, please contact the corresponding author at [zhxj@qhnu.edu.cn](mailto:zhxj@qhnu.edu.cn).
- Modify the path to the training dataset in `src/config/ds/train.yaml`.
- Modify the path to the validation dataset in `src/config/ds/valid.yaml` (if have any).
- Run the following command:
    ```
    python src/train.py name=your_experiment_name
    ```
- The trained models and intermediate results will be saved in the `log/` folder.

#### OOM Errors

You may need to reduce the batch size in `src/config/config.yaml` to avoid out of memory errors. If you do this, but want to preserve quality, be sure to increase the number of training iterations and decrease the learning rate by whatever scale factor you decrease batch size by.
Should you have any questions, feel free to post an issue or contact me at [zhxj@qhnu.edu.cn](mailto:zhxj@qhnu.edu.cn).

## 📜 Citation Request 
```bibtex
@article{thangka2025tvc,
  title = {Illumination Separation and Adaptive Color Compensation: Enhancing Thangka Image Exposure Correction},
  author = {WU Ji-yuan and ZHANG Xiao-juan},
  journal = {Submitted to The Visual Computer},
  year = {2025},
  note = {Submission ID: }
}

