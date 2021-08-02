-The main fork did not have the train code. I have added training file. Used loss is MSE different than the main fork. 

-The main fork did not have the pathing the input images I have added a patch.py to patch input images for training procces 

-The main for has not visualized intermediate layers. I have visualized them and saved them in u1, d64u1, d4u1 files. If you want to visualize your input data you can uncooment line 276 in test file.

-The main fork was using old environment setup, I have integrated the codes so that you can use them with latedt cuda, python and ubuntu versions.

# Trident Dehazing Network
NTIRE 2020 NonHomogeneous Dehazing Challenge (CVPR Workshop 2020)  **1st** Solution.

[[Challenge report]]( https://arxiv.org/pdf/2005.03457.pdf )
[[TDN paper]]( http://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Liu_Trident_Dehazing_Network_CVPRW_2020_paper.pdf )

### Environment:

- Ubuntu 20.04
- Python3.8.10
- NVIDIA GPU+CUDA10.1

### Dependencies:

- pretrainedmodels==0.7.4
- torchvision==0.9.1
- torch==1.8.1
- tqdm

### Test

Compile the DCN module fisrt. If your environment is the same as ours, compile was done. If your pytorch version is 1.0.0, use DCNv2_pytorch1.

Check the hazy images path (test.py line 14), the model path (test.py line 13) and the output path (test.py line 15)

```
python test.py --inp <hazyImageFolder>
```

### Train
dont forget to check the input file
```
python train.py
```

### Pretrained model

https://pan.baidu.com/s/1l0-hOnIAAbFzmauUmFaRjw  password: 22so

https://drive.google.com/file/d/1LcSsCWGLkjmq5o08yhMbSU6DjCGugmRw


