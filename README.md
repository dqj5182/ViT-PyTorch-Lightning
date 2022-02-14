# SNU_CV_VIT_CIFAR

Folder Structure
```bash
.
├── dataset
│   └── load_cifar.py
├── model
│   ├── HCGNet
│   ├── densenet
│   ├── dla
│   ├── dpn
│   ├── efficientnet
│   ├── efficientnetV2
│   ├── mobilenetV3
│   ├── resnet
│   ├── resnext
│   └── vgg
├── notebooks
│   └── Pretrained_HCGNet.ipynb
├── main.py
├── train_model.py
├── test_model.py             
└── utils.py            
```

Train the model with CIFAR-10 dataset:
```
python main.py
```

</br>

## Pretrained Models
Epoch: 350
Model | Pretrained (.pth) | Paper | Accuracy | Architecture
--- | --- | --- | --- | ---
HCGNet | [Google Drive](https://drive.google.com/file/d/11SvHuhBjHElmlp80dIJn0AokisiewLNd/view?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1908.09699.pdf) | 91.92% | hybrid (dense, residual) connectivity, micro-module and attention-based forget and update gates
DenseNet | [Google Drive](https://drive.google.com/file/d/14-y22orjDvQJBPm6vYiMnDP3FLYYzAcH/view?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1608.06993.pdf) | 90.18% | identity mappings, deep supervision, and diversified depth
PyramidNet | [Google Drive](https://drive.google.com/file/d/1Ln7e7n6KIN8xYTBPHR43H_Uh07AEt7LC/view?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1610.02915.pdf) | 89.89% | increasing the feature map dimension gradually
ResNeXT | [Google Drive](https://drive.google.com/file/d/1HAS2WeF1i8IwaiRMFA_xFBqmi6fAlip4/view?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1611.05431.pdf) | 89.51% | aggregates a set of transformations with the same topology
DPN | [Google Drive](https://drive.google.com/file/d/1W6EZ-caNd9N6m7eIRzl-cx5N2RUDUh14/view?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1707.01629.pdf) | 89.25% | integrate feature re-usage (ResNet) and new features exploration (DenseNet)

</br>

## Codes with accuracy above 95%

These are some of the codes (open to public) that achieve accuracy of above 95%</br>
* **LaMCTS (Latent Action Monte Carlo Tree Search)**</br>
  * [GitHub](https://github.com/facebookresearch/LaMCTS)</br>
  * CIFAR-10 accuracy: **99.03%**</br>
* **Divide and Co-training**</br>
  * [GitHub](https://github.com/mzhaoshuai/Divide-and-Co-training)</br>
  * CIFAR-10 accuracy: **98.71%**</br>
* **MPL (Meta Pseudo Labels)**</br>
  * [GitHub](https://github.com/kekmodel/MPL-pytorch)</br>
  * CIFAR-10 accuracy: **96.11%**</br>

## Reference
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html</br>
https://github.com/lukemelas/EfficientNet-PyTorch</br>
https://github.com/facebookresearch/LaMCTS</br>
https://github.com/kekmodel/MPL-pytorch</br>
https://github.com/MadryLab/cifar10_challenge</br>
https://github.com/xiaolai-sqlai/mobilenetv3</br>
https://github.com/EN10/CIFAR</br>
https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54</br>
