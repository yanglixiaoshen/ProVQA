# Blind visual quality assessment on omnidirectional or 360 videos (ProVQA)

Blind VQA for 360° Video via Progressive Learning from Pixels, Frames and Video (Under review of a IEEE transaction)


This repository contains the official PyTorch implementation of the following paper:
> **Blind VQA for 360° Video via Progressive Learning from Pixels, Frames and Video (IEEE xxxx)**<br>
> Li Yang, Mai Xu, YiChen Guo and ShengXi Li (School of Electronic and Information Engineering, Beihang University)<br>
> **Paper link**: https://xxx. <br>
> **Abstract**: *xxxxxx.*

## Preparation

### Requriments 

First, install my conda enviroment \<envs\> in Linux sys (Ubuntu 18.04+); Then, activate \<envs\> by running the following command:
```shell
conda env create -f ProVQA_environment.yaml
```
Second, we install all dependency by running the following command:
```shell
pip install -r ProVQA_environment.txt
```

## Implementation

The architecture of the proposed SAP-net is shown in the following figure, which contains three novel modules, i.e., WBRE, PQE and QR.

<div align="center"><img width="93%" src="https://github.com/yanglixiaoshen/ProVQA/blob/main/images/framework6.png" /></div>



