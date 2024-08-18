# MSFINet
The official implementation code of "MSFINet: Multi-scale Feature Interaction Fusion Network for Skin Lesion Image Segmentation". We will continue to improve the relevant content.
## Framework Overview
![Framework Overview](https://github.com/WikYue/MSFINet/tree/main/figs/MSFINet.jpg)

## Experimental results
### Comparison of the number of parameters and computation of the network
 Model  | Flops(G)  | Params(M)
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
   | ISIC2017 | ISIC2018
   | DSC(%) | ACC(%) | DSC(%) | ACC(%)
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
