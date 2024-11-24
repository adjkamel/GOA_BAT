This repository contains code for the paper:  [3D Human Pose Estimation via Spatial Graph Order
Attention and Temporal Body Aware Transformer
](https://) 

## Environment

* Python 3.8.2
* PyTorch 3.9.16
* CUDA 12.0

## Datasets
We follow the links provided by [PoseFormer](https://github.com/zczcwh/PoseFormer), [GLA-GCN](https://github.com/bruceyo/GLA-GCN), and [P-STMO](https://github.com/patrick-swk/p-stmo) 
- Human3.6m: [CPN 2D](https://drive.google.com/file/d/1ayw5DI-CwD4XGtAu69bmbKVOteDFJhH5/view), [Ground_truth 2D](https://drive.google.com/file/d/1U0Z85HBXutOXKMNOGks4I1ape8hZsAMl/view), [Ground_truth 3D](https://drive.google.com/file/d/13PgVNC-eDkEFoHDHooUGGmlVmOP-ri09/view)
- HumanEva-I: [Ground_truth 2D](https://drive.google.com/file/d/1UuW6iTdceNvhjEY2rFF9mzW93Fi1gMtz/view),  [Ground_truth 3D](https://drive.google.com/file/d/1CtAJR_wTwfh4rEjQKKmABunkyQrvZ6tu/view)  
- MPI-INF-3DHP: [Train and Test](https://drive.google.com/file/d/11eBe175Rgj6IYrwZwa1oXTOyHPxGuWyi/view)

Please put them in folder ./data to reproduce the results. 

## Evaluating pre-trained models

We provide the pre-trained [models](https://drive.google.com/drive/folders/1Lkjr95nv3gDCsLlVcgwfmYZxRDrbLI1p?usp=drive_link) using CPN and GT 2D data, please put them in ./checkpoint directory. To evaluate, pleasae run:

### Human3.6M
- On ground truth 2D
```
main_h36m.py -k gt -f 324 --evaluate h36m_gt_best.bin
```
-On CPN 2D
```
python main_h36m.py -k cpn_ft_h36m_dbb -f 324 --evaluate h36m_cpn_best.bin
```

### HumanEva-I
```
python main_heva.py -k gt -f 5 --evaluate heva1_gt_best.bin
```
### MPI-INF-3DHP
```
python main_3dhp.py -f 81 -frame-kept 9 -coeff-kept 9 --reload 1 --previous_dir checkpoint/best_mpi-inf-3dhp.pth
```
## Training new models

### Human3.6M
-On ground truth 2D
```
python main_h36m.py -k gt -f 324 -lr 0.0003 -lrd 0.95
```
-On CPN 2D
```
python main_h36m.py -k cpn_ft_h36m_dbb -f 324 -lr 0.0003 -lrd 0.95
```
### HumanEva-I
```
python main_heva.py -k gt -f 5 -lr 0.001 -lrd 0.95 
```
### MPI-INF-3DHP
```
python main_3dhp.py -f 81 -frame-kept 9 -coeff-kept 9 -b 512 --train 1 --lr 0.0007 -lrd 0.97 -c CKPT_NAME --gpu 1
```

