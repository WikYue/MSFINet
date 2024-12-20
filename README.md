# MSFINet
The official implementation code of "MSFINet: Multi-scale Feature Interaction Fusion Network for Skin Lesion Image Segmentation". We will continue to improve the relevant content.
## Framework Overview
![Framework Overview](https://github.com/WikYue/MSFINet/blob/main/configs/MSFINet.jpg)

## Experimental results
### Table1 Comparison of the number of parameters and computation of the network
 Model | Flops(G) | Params(M)
 ---- | ----- | ------  
 EGE-UNet  | 0.07 | 0.05 
 MALU-Net  | 0.08 | 0.17 
 UNet  | 11.92 | 22.53
 CSCAUNet  | 19.23 | 35.27
 DCSAU-Net  | 21.17 | 25.99
 MedT  | 17.26 | 61.53
 ---- | ----- | ------ 
 TransNorm  | 17.29 | 31.17
 CTO  | 22.72 | 59.82
 FAT-Net  | 23.73 | 30.98
 DHUNet  | 101.07 | 69.34
 MSFINet(Ours)  | 11.79 | 47.29

### Table2 Impact of different loss function settings on model performance
 | | ISIC2017 | |ISIC2018| |
 ---- | ----- | ------ | ------ | ------ |
 | | DSC(%) | ACC(%) | DSC(%) | ACC(%)
 *L<sub>bce</sub>* | 90.81 | 96.73 | 90.39 | 94.22
 *L<sub>dice</sub>* | 90.49 | 96.46 | 89.97 | 94.08
 *L<sub>bce</sub> + L<sub>dice</sub>* | 91.17 | 96.97 | 90.37 | 94.92
 *λ<sub>0</sub>, λ<sub>1</sub>, λ<sub>2</sub>* = 1, 1, 1 | 90.63 | 96.62 | 90.44 | 94.35
 *λ<sub>0</sub>, λ<sub>1</sub>, λ<sub>2</sub>* = 1, 0.1, 0.1 | 90.02 | 96.28 | 90.25 | 94.13
 *λ<sub>0</sub>, λ<sub>1</sub>, λ<sub>2</sub>* = 1, 0.5, 0.5 | 91.17 | 96.97 | 90.37 | 94.92

### Table3 Ablation experiments with different modules
 | | | | | | ISIC2017 | |ISIC2018| |
 | ---- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
 | Model | CIT | CRA | MKDA | BDO | DSC(%) | ACC(%) | DSC(%) | ACC(%)
 | Baseline |  |  |  |  | 87.39 | 95.14 | 86.02 | 93.16
 |  | √ |  |  |  |88.07 | 95.56 | 87.84 | 93.53
 |  |  | √ |  |  |88.51 | 95.89 | 88.27 | 93.67
 |  |  |  | √ |  |87.83 | 95.48 | 88.15 | 93.64
 |  |  |  |  | √ |87.46 | 95.22 | 87.73 | 93.46
 |  | √ | √ |  |  |89.56 | 96.41 | 88.72 | 93.63
 |  |  |  | √ | √ |89.26 | 96.20 | 87.71 | 93.43
 |  | √ |  | √ | √ |90.82 | 96.56 | 89.28 | 93.97
 |  |  | √ | √ | √ |89.39 | 96.09 | 87.92 | 93.69
 |  | √ | √ | √ |  |90.39 | 96.58 | 89.95 | 93.98
 |  | √ | √ |  | √ | 90.03 | 96.42 | 90.19 | 94.10
 |  | √ | √ | √ | √ | 91.17 | 96.97 | 90.37 | 94.92

### Table4 Impact of Hyperparameter Settings on Network Performance
 | | ISIC2017 | |ISIC2018| |
 ---- | ----- | ------ | ------ | ------ |
 Hyperparameter | DSC(%) | ACC(%) | DSC(%) | ACC(%)
 lr = 0.001 | 91.17 | 96.97 | 90.37 | 94.92
 lr = 0.005 | 89.83 | 95.97 | 88.27 | 93.64
 lr = 0.01 | 88.02 | 95.41 | 87.15 | 92.82
 batch_size = 8 | 90.78 | 96.69 | 90.03 | 94.26
 batch_size = 16 | 91.17 | 96.97 | 90.37 | 94.92
 batch_size = 32 | 90.12 | 96.58 | 90.22 | 94.83


 
