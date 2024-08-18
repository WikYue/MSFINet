# MSFINet
The official implementation code of "MSFINet: Multi-scale Feature Interaction Fusion Network for Skin Lesion Image Segmentation". We will continue to improve the relevant content.
## Framework Overview
![Framework Overview](https://github.com/WikYue/MSFINet/tree/main/figs/MSFINet.jpg)

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
 Loss function | DSC(%) | ACC(%) | DSC(%) | ACC(%)
 *L<sub>bce</sub>* | 90.81 | 96.73 | 90.39 | 94.22
 *L<sub>dice</sub>* | 90.49 | 96.46 | 89.97 | 94.08
 *L<sub>bce</sub> + L<sub>dice</sub>* | 91.71 | 96.97 | 92.37 | 95.92
 *λ<sub>0</sub>, λ<sub>1</sub>, λ<sub>2</sub>* = 1, 1, 1 | 90.63 | 96.62 | 90.44 | 94.35
 *λ<sub>0</sub>, λ<sub>1</sub>, λ<sub>2</sub>* = 1, 0.1, 0.1 | 90.02 | 96.28 | 90.25 | 94.13
 *λ<sub>0</sub>, λ<sub>1</sub>, λ<sub>2</sub>* = 1, 0.5, 0.5 | 91.71 | 96.97 | 92.37 | 95.92

### Table3 Ablation experiments with different modules
 | | | | | | ISIC2017 | |ISIC2018| |
 | ---- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
 




 
