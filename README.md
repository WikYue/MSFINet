# MSFINet
The official implementation code of "MSFINet: Multi-scale Feature Interaction Fusion Network for Skin Lesion Image Segmentation". We will continue to improve the relevant content.
## Framework Overview
![Framework Overview](https://github.com/WikYue/MSFINet/tree/main/figs/MSFINet.jpg)

## Experimental results
### Comparison of the number of parameters and computation of the network
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

### Comparison of the number of parameters and computation of the network
 | | ISIC2017 | |ISIC2018| |
 ---- | ----- | ------ | ------ | ------
 Hyperparameter | DSC(%) | ACC(%) | DSC(%) | ACC(%)
 ---- | ----- | ------
L_bce | 90.81 | 96.73 | 90.39 | 94.22
L_dice | 90.49 | 96.46 | 89.97 | 94.08
L_bce+L_dice | 91.71 | 96.97 | 92.37 | 95.92
λ_0,λ_1,λ_2=1,1,1 | 90.63 | 96.62 | 90.44 | 94.35
λ_0,λ_1,λ_2=1,0.1,0.1 | 90.02 | 96.28 | 90.25 | 94.13
λ_0,λ_1,λ_2=1,0.5,0.5 | 91.71 | 96.97 | 92.37 | 95.92
