# Blind visual quality assessment on omnidirectional or 360 video (ProVQA)

Blind VQA for 360° Video via Progressively Learning from Pixels, Frames and Video


This repository contains the official PyTorch implementation of the following paper:
> **Blind VQA for 360° Video via Progressive Learning from Pixels, Frames and Video**<br>
> Li Yang, Mai Xu, ShengXi Li, YiChen Guo and Zulin Wang  (School of Electronic and Information Engineering, Beihang University)<br>
> **Paper link**: xxxxxx. <br>
> **Abstract**: *Blind visual quality assessment (BVQA) on 360° video plays a key role in optimizing immersive multimedia systems. When assessing the quality of 360° video, human tends to perceive its quality degradation from the viewport-based spatial distortion of each spherical frame to motion artifact across adjacent frames, ending with the video-level quality score, i.e., a progressive quality assessment paradigm. However, the existing BVQA approaches for 360° video neglect this paradigm. In this paper, we take into account the progressive paradigm of human perception towards spherical video quality, and thus propose a novel BVQA approach (namely ProVQA) for 360° video via progressively learning from pixels, frames and video. Corresponding to the progressive learning of pixels, frames and video, three sub-nets are designed in our ProVQA approach, i.e., the spherical perception aware quality prediction (SPAQ), motion perception aware quality prediction (MPAQ) and multi-frame temporal non-local (MFTN) sub-nets. The SPAQ sub-net first models the spatial quality degradation based on spherical  perception  mechanism of human. Then, by exploiting motion cues across adjacent frames, the MPAQ sub-net properly incorporates motion contextual information for quality assessment on 360° video. Finally, the MFTN sub-net aggregates multi-frame quality degradation to yield the final quality score, via exploring long-term quality correlation from multiple frames. The experiments validate that our approach significantly advances the state-of-the-art BVQA performance on 360° video over two datasets, the code of which has been public in \url{https://github.com/yanglixiaoshen/ProVQA.}* <br>
> **Note: Since this paper is under review, you can first ask for the paper from me to ease the implementation of this project but you have no rights to use this paper
> in any purpose. Unauthorized use of this article for all activities will be investigated for legal responsibility. Contact me for accessing my paper (Email: 13021041@buaa.edu.cn)**

## Preparation

### Requriments 

First, download the conda environment of ProVQA from [ProVQA_dependency](https://www.dropbox.com/sh/4iybfi77llpu55p/AAAAFNLDqNI9ptit7YMYquAya?dl=0) and install my conda enviroment \<envs\> in Linux sys (Ubuntu 18.04+); Then, activate \<envs\> by running the following command:
```shell
conda env create -f ProVQA_environment.yaml
```
Second, install all dependencies by running the following command:
```shell
pip install -r ProVQA_environment.txt
```
If the above installation don't work, you can download the [environment file](https://www.dropbox.com/s/eu429a39e6tlzyi/bvqa.tar.gz?dl=0) with .tar.gz format. Then, unzip the file into a directory (e.g., pro_env) in your home directiory and activate the environment every time before you run the code.

```shell
source activate /home/xxx/pro_env
```


## Implementation

The architecture of the proposed ProVQA is shown in the following figure, which contains four novel modules, i.e., SPAQ, MPAQ, MFTN and AQR.

<div align="center"><img width="93%" src="https://github.com/yanglixiaoshen/ProVQA/blob/main/images/framework6.png" /></div>

### Dataset

We trained our ProVQA on the large-scale 360° VQA dataset [VQA-ODV](https://github.com/Archer-Tatsu/VQA-ODV), which includes 540 impaired 360° videos deriving 
from 60 reference 360° videos under equi-rectangular projection (ERP) (Training set: 432-Testing set:108). Besides, we also evaluate the performance of our ProVQA
over 144 distorted 360° videos in [BIT360](https://ieeexplore.ieee.org/document/8026226) dataset.

### Training the ProVQA

Our network is implemented based on the PyTorch framework, and run on two NVIDIA Tesla V100 GPUs with 32G memory. The number of 
sampled frames is 6 and the batch size is 3 per GPU for each iteration. The training set of VQA-ODV dataset has been packed as an LMDB file [ODV-VQA_Train](https://www.dropbox.com/sh/1a7hn3fhwbe1lj5/AACekKngkO2Y2hgVzWxv8Wqca?dl=0), which 
is used in our approach.

First, to run the training code as follows,

```shell

CUDA_VISIBLE_DEVICES=0,1 python ./train.py -opt ./options/train/bvqa360_240hz.yaml

```
Note that all the settings of dataset, training implementation and network can be found in "bvqa360_240hz.yaml". You can modify the settings to
satisfy your experimental environment, for example, the dataset path should be modified to be your sever path. 
For the final BVQA result, we choose the trained model at iter=26400, which can be download at [saved_model](https://www.dropbox.com/s/jxlps73yadwixr0/net_g_26400.pth?dl=0). Moreover, 
the corresponding training state can be obtained at [saved_optimized_state](https://www.dropbox.com/s/g2favdkl2s1dbys/26400.state?dl=0).

### Testing the ProVQA

Download the [saved_model](https://www.dropbox.com/s/jxlps73yadwixr0/net_g_26400.pth?dl=0) and put it in your own experimental
directory. Then, run the following code for evaluating the BVQA performance over the testing set [ODV-VQA_TEST](https://www.dropbox.com/s/6hd96pfxg1yflgd/data_all_test_png.zip?dl=0).
Note that all the settings of testing set, testing implementation and results can be found in "test_bvqa360_OURs.yaml". You can modify the settings to satisfy your experimental environment.


```shell

CUDA_VISIBLE_DEVICES=0 python ./test.py -opt ./options/test/test_bvqa360_OURs.yaml

```
The test results of predicted quality scores of all test 360° Video frames can be found in [All_frame_scores](https://www.dropbox.com/s/d01r36i7336tp9t/BVQA360_dmos240_26400.txt?dl=0) 
and latter you should run the following code to generate the final 108 scores corresponding to 108 test 360° Videos, which can be downloaded from [predicted_DMOS](https://www.dropbox.com/s/722lq4h5bfqk2lv/ours_dmos.txt?dl=0).

```shell

python ./evaluate.py

```


## Evaluate BVQA performance

We have evaluate the BVQA performance for 360° Videos by 5 general metrics: PLCC, SROCC, KROCC, RMSE and MAE.
we employ a 4-order logistic function for fitting the predicted quality scores to their corresponding ground truth, 
such that the fitted scores have the same scale as the ground truth DMOS [gt_dmos](https://www.dropbox.com/s/b88b3p8k7w7n7dw/gt_dmos.txt?dl=0). Note that the fitting procedure are conducted on our and all compared approaches.
Run the code [bvqa360_metric](https://www.dropbox.com/s/qfu1n4uua82t843/bvqa360_metric.m?dl=0) in the following command :

```shell

./bvqa360_metric.m

```
As such, you can get the final results of PLCC=0.9209, SROCC=0.9236, KROCC=0.7760, RMSE=4.6165 and MAE=3.1136.
The following tables shows the comparison on BVQA performance between our and other 13 approaches, over VQA-ODV and BIT360 dataset.

<div align="center"><img width="93%" src="https://github.com/yanglixiaoshen/ProVQA/blob/main/images/t1.png" /></div>

<div align="center"><img width="45%" src="https://github.com/yanglixiaoshen/ProVQA/blob/main/images/t2.png" /></div>

## Tips

>**(1) We have summarized the information about how to run the compared algorithms in details, which can be found in the file "compareAlgoPreparation.txt".**<br>
>**(2) The details about the pre-processing on the ODV-VQA dataset and BIT360 dataset can be found in the file "pre_process_dataset.py".** <br>



## Citation 

If this repository can offer you help in your research, please cite the paper:

It will be released in the near future. 

Please enjoy it and best wishes. Plese contact with me if you have any questions about the ProVQA approach.

My email address is 13021041[at]buaa[dot]edu[dot]cn
