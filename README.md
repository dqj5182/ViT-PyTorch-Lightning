# SNU_CV_VIT_CIFAR

Folder Structure
```bash
.
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
│   ├── vgg
│   └── vit
├── notebooks
│   └── Pretrained ResNet (VIT Week).ipynb
├── utils
│   ├── autoaugment.py
│   ├── dataaug.py
│   └── utils.py
├── weights
├── main.py
└── setup.sh           
```

Train the model with CIFAR-10 dataset:
```
python main.py --dataset c10 --label-smoothing --autoaugment --model-name [name of the model]
```

</br>

## Pretrained Models
Epoch: 350
Model | Pretrained (.pth) | Notebook | Paper | Accuracy
--- | --- | --- | ---
ResNet | [Google Drive](https://drive.google.com/file/d/1FNsXhHEpEKiZQFgk8vBppmGIABEYXb88/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/1NSBEJSnQ4wgt6_bvPPKlbyxASZcZgfH6?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1512.03385.pdf) | 95.36%

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
