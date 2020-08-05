# FastV2C-HandNet : Fast Voxel to Coordinate Hand Pose Estimation with 3D Convolutional Neural Networks

# Introduction

This is the project repository for the paper, **FastV2C-HandNet : Fast Voxel to Coordinate Hand Pose Estimation with 3D Convolutional Neural Networks ([Springer](https://link.springer.com/chapter/10.1007/978-981-15-5113-0_31))**.


Please refer to our paper for details.

If you find our work useful in your research or publication, please cite our work:

[1] Rohan Lekhwani, Bhupendra Singh. **"FastV2C-HandNet : Fast Voxel to Coordinate Hand Pose Estimation with 3D Convolutional Neural Networks"**[[Springer](https://link.springer.com/chapter/10.1007/978-981-15-5113-0_31)]

  ```
Lekhwani, Rohan, and Bhupendra Singh. 
"FastV2C-HandNet: Fast Voxel to Coordinate Hand Pose Estimation with 3D Convolutional Neural Networks." 
International Conference on Innovative Computing and Communications. 
Springer, Singapore, 2019.
  }
```

In this repository, we provide
* Our model architecture description (FastV2C-HandNet)
* Comparison with the previous state-of-the-art methods
* Training code
* Dataset we used (MSRA)
* Trained models and estimated results

## Model Architecture

![FastV2C-HandNet](/figs/Figure_3.png)

## Comparison with the previous state-of-the-art methods

![Paper_result_hand_table](/figs/Table_1.png)

![Paper_result_v2v-posenet_table](/figs/Table_2.png)

# About our code
## Dependencies
* [Keras](http://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)

The code is tested under Ubuntu 18.04, Windows 10 environment with Nvidia P100 GPU (16GB VRAM).

## Code
Clone this repository into any place you want. You may follow the example below.
```bash
makeReposit = [/the/directory/as/you/wish]
mkdir -p $makeReposit/; cd $makeReposit/
git clone https://github.com/RonLek/FastV2C-HandNet.git
```
* `src` folder contains python script files for data loader, trainer, tester and other utilities.
* `data` folder should contain an 'MSRA' folder with binary image files.

To train our model, please run the following command in the `src` directory:

```bash
python train.py
```
# Dataset
We trained and tested our model on the MSRA Hand Pose Dataset.

* MSRA Hand Pose Dataset [[link](https://jimmysuen.github.io/)] [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf)]

# Results
Here we provide the precomputed centers, estimated 3D coordinates and pre-trained models of MSRA dataset. You can download precomputed centers and 3D hand pose results in [here](/results/centers) and pre-trained models in [here](/results/checkpoints/model3)

The precomputed centers are obtained by training the hand center estimation network from [DeepPrior++ ](https://arxiv.org/pdf/1708.08325.pdf). Each line represents 3D world coordinate of each frame.
In case depth map does not exist or not contain hand, that frame is considered as invalid.
All test images are considered as valid.

We used [awesome-hand-pose-estimation ](https://github.com/xinghaochen/awesome-hand-pose-estimation) to evaluate the accuracy of the FastV2C-HandNet on the MSRA dataset.

Belows are qualitative results.
![result_1](/figs/Figure_4.png)
![result_2](/figs/Figure_5.png)
