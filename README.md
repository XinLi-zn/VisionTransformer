# Awesome Transformer in Vision [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)  
A curated list of vision transformer related resources. Please feel free to [pull requests](https://github.com/penghouwen/VisionTransformer/pulls) or [open an issue](https://github.com/penghouwen/VisionTransformer/issues) to add papers.


## Table of Contents

- [Awesome Surveys](#awesome-surveys)

- [Transformer in Vision](#transformer-in-vision)
  - [2021 Venues](#2021)
  - [2020 Venues](#2020)
  - [2019 Venues](#2019)
  - [Previous Venues](#2012-2018)
  
  - [Awesome Libraies](#awesome-surveys)

## Awesome Surveys

|  Title  |   Venue  |   BibTeX  |
|:--------|:--------:|:--------:|
| [A Survey on Visual Transformer](https://arxiv.org/pdf/2012.12556.pdf) | ArXiv | [Bib](https://scholar.googleusercontent.com/scholar.bib?q=info:Aj10Crv7DScJ:scholar.google.com/&output=citation&scisdr=CgUmooQTEM3KnAOogfQ:AAGBfm0AAAAAX_-tmfT1yhaAeO62lS61HGcSpcXSUqJ5&scisig=AAGBfm0AAAAAX_-tmQAIcm-VKBRqnb9iTs8Sghq-6ssB&scisf=4&ct=citation&cd=-1&hl=ja)
|[Intriguing Properties of Vision Transformers](https://arxiv.org/pdf/2105.10497.pdf)| ArXiv|[Code](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers)|
|[CVPR 2021 视觉Transformer论文（43篇）](https://mp.weixin.qq.com/s/6G9vHSwURwlT2oH312Opww)| github|--|


## Transformer in Vision

|      Task   |        Reg       |       Det    |           Seg           |        Trk           |    Other   |
|:------------|:--------------:|:----------------------:|:-----------------------:|:----------------------:|:----------:|
| Explanation | Image Recoginition | Object Detection | Image Segmentation | Object Tracking | other types |

You can add a tag for `domains` which contains several transformer-based works

### 2021
(Pls follow Time Inverse Ranking)

|  Title  |   Venue  |  Task  |   Code   |  BibTeX  |
|:--------|:--------:|:--------:|:--------:|:--------:|
|[Tracking Instances as Queries](https://arxiv.org/pdf/2106.11963.pdf)|arxiv|Seg|--|--|
|[Instances as Queries](https://arxiv.org/pdf/2105.01928)|arxiv|Seg|--|[GitHub](https://github.com/hustvl/QueryInst)|
|[OadTR: Online Action Detection with Transformers](https://arxiv.org/pdf/2106.11149.pdf)|CVPRW|Det|--|[GitHub](https://github.com/wangxiang1230/OadTR)|
|[An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)|ArXiv|Other|--|--|
|[End-to-end Temporal Action Detection with Transformer](https://arxiv.org/pdf/2106.10271.pdf)|ArXiv|Cls|--|[GitHub](https://github.com/xlliu7/TadTR)|
|[MlTr: Multi-label Classification with Transformer](https://arxiv.org/pdf/2106.06195.pdf)|ArXiv|Cls|--|[GitHub](https://github.com/starmemda/MlTr/)|
|[Delving Deep into the Generalization of Vision Transformers under Distribution Shifts](https://arxiv.org/pdf/2106.07617.pdf)|ArXiv|Other|--|--|
|[Improved Transformer for High-Resolution GANs](https://arxiv.org/pdf/2106.07631.pdf)|ArXiv|Other|--|--|
|[BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf)|ArXiv|Cls|--|[GitHub](https://github.com/ZhendongWang6/Uformer)|
|[XCiT: Cross-Covariance Image Transformers](https://arxiv.org/pdf/2106.09681.pdf)|ArXiv|Other|--|--|
|[Semi-Autoregressive Transformer for Image Captioning](https://arxiv.org/pdf/2106.09436.pdf)|ArXiv|Other|--|--|
|[Long-Short Temporal Contrastive Learning of Video Transformers](https://arxiv.org/pdf/2106.09212.pdf)|ArXiv|Other|--|--|
|[Uformer: A General U-Shaped Transformer for Image Restoration](https://arxiv.org/abs/2106.03106)|ArXiv|Other|--|[GitHub](https://github.com/microsoft/unilm/tree/master/beit)|
|[Video Super-Resolution Transformer](https://arxiv.org/abs/2106.06847)|ArXiv|Other|--|[GitHub](https://github.com/caojiezhang/VSR-Transformer)|
|[DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://arxiv.org/abs/2106.02034)|ArXiv|Cls|--|[GitHub](https://github.com/raoyongming/DynamicViT)|
|[Semantic Correspondence with Transformers](https://arxiv.org/pdf/2106.02520.pdf)|ArXiv|Other|--|[GitHub](https://github.com/SunghwanHong/CATs)|
|[Glance-and-Gaze Vision Transformer](https://arxiv.org/pdf/2106.02277.pdf)|ArXiv|Other|--|[GitHub](https://github.com/yucornetto/GG-Transformer)|
|[Few-Shot Segmentation via Cycle-Consistent Transformer](https://arxiv.org/pdf/2106.02320.pdf)|ArXiv|Seg|--|--|
|[Self-Supervised Learning with Swin Transformers](https://arxiv.org/pdf/2105.04553.pdf)|ArXiv|Other|--|[GitHub](https://github.com/SwinTransformer/Transformer-SSL)|
|[Visual Grounding with Transformers](https://arxiv.org/pdf/2105.04281.pdf)|ArXiv|Other|--|--|
|[Associating Objects with Transformers for Video Object Segmentation](https://arxiv.org/pdf/2106.02638.pdf)|ArXiv|Seg|--|[GitHub](https://github.com/z-x-yang/AOT)|
|[When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations](https://arxiv.org/pdf/2106.01548.pdf)|ArXiv|Other|--|--|
|[DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://arxiv.org/pdf/2106.02034.pdf)|ArXiv|Other|--|[GitHub](https://github.com/raoyongming/DynamicViT)|
|[Anticipative Video Transformer](https://arxiv.org/pdf/2106.02036.pdf)|ArXiv|Other|--|[GitHub](https://facebookresearch.github.io/AVT/)|
|[An Attention Free Transformer](https://arxiv.org/pdf/2105.14103.pdf)|ArXiv|Other|--|--|
|[Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks](https://arxiv.org/abs/2105.02358)|ArXiv|Other|[GitHub](https://github.com/MenghaoGuo/-EANet)|--|
|[TransVOS: Video Object Segmentation with Transformers](https://arxiv.org/pdf/2106.00588.pdf)|ArXiv|Seg|--|--|
|[You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/pdf/2106.00666.pdf)|ArXiv|Det|[GitHub](https://github.com/hustvl/YOLOS)|--|
|[ResT: An Efficient Transformer for Visual Recognition](https://arxiv.org/pdf/2105.13677.pdf)|ArXiv|Reg|[GitHub](https://github.com/wofmanaf/ResT)|--|
|[Not All Images are Worth 16x16 Words: Dynamic Vision Transformers with Adaptive Sequence Length](https://arxiv.org/pdf/2105.15075)|ArXiv|Other|--|--|
|[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/pdf/2105.15203.pdf)|ArXiv|Seg|--|--|
|[Aggregating Nested Transformers](https://arxiv.org/pdf/2105.12723.pdf)|ArXiv|Other|--|--|
|[End-to-End Video Object Detection with Spatial-Temporal Transformers](https://arxiv.org/pdf/2105.10920.pdf)|ArXiv|Det|[GitHub](https://github.com/SJTU-LuHe/TransVOD)|--|
|[HOTR: End-to-End Human-Object Interaction Detection with Transformers](https://arxiv.org/pdf/2101.01909)|CVPR2021|Other|[GitHub](https://github.com/mlpc-ucsd/LETR)|--|
|[Line Segment Detection Using Transformers without Edges](https://arxiv.org/pdf/2104.13682.pdf)|CVPR2021|Other|--|--|
|[Boosting Crowd Counting with Transformers](https://arxiv.org/pdf/2105.10926.pdf)|ArXiv|Other|--|--|
|[Points as Queries: Weakly Semi-supervised Object Detection by Points](https://arxiv.org/pdf/2104.07434.pdf)|ArXiv|Other|--|--|
| [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986) | Arxiv | Reg | [GitHub](https://github.com/yitu-opensource/T2T-ViT) | <details> <summary>Bib</summary> <p align="left">   </br> @article{yuan2021tokens, </br> title={Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet}, </br> author={Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Tay, Francis EH and Feng, Jiashi and Yan, Shuicheng}, </br> journal={arXiv preprint arXiv:2101.11986}, </br> year={2021} </br> } </p></details> </br>
| [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605) | Arxiv | Reg | [GitHub](https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2) | <details> <summary>Bib</summary> <p align="left">   </br> @article{srinivas2021bottleneck, </br> title={Bottleneck Transformers for Visual Recognition}, </br> author={Srinivas, Aravind and Lin, Tsung-Yi and Parmar, Niki and Shlens, Jonathon and Abbeel, Pieter and Vaswani, Ashish}, </br> journal={arXiv preprint arXiv:2101.11605}, </br> year={2021} </br> } </p></details> </br>
| [SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation](https://arxiv.org/abs/2101.08833) | Arxiv | Seg | --- | <details> <summary>Bib</summary> <p align="left">   </br> @article{duke2021sstvos, </br> title={SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation}, </br> author={Duke, Brendan and Ahmed, Abdalla and Wolf, Christian and Aarabi, Parham and Taylor, Graham W}, </br> journal={arXiv preprint arXiv:2101.08833}, </br> year={2021} </br> } </p></details> </br>
| [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702) | Arxiv | Trk | --- | <details> <summary>Bib</summary> <p align="left">   </br> @article{meinhardt2021trackformer, </br> title={TrackFormer: Multi-Object Tracking with Transformers}, </br> author={Meinhardt, Tim and Kirillov, Alexander and Leal-Taixe, Laura and Feichtenhofer, Christoph}, </br> journal={arXiv preprint arXiv:2101.02702}, </br> year={2021} </br> } </p></details> </br>


### 2020

|  Title  |   Venue  |  Task  |   Code   |  BibTeX  |
|:--------|:--------:|:--------:|:--------:|:--------:|
|[End-to-End Video Instance Segmentation with Transformers](https://arxiv.org/pdf/2011.14503.pdf)|ArXiv|Seg|--|--|
| [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) | ArXiv | Reg | [GitHub](https://github.com/facebookresearch/deit) | <details> <summary>Bib</summary> <p align="left">   </br> @article{touvron2020training, </br> title={Training data-efficient image transformers \& distillation through attention}, </br> author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}}, </br> journal={arXiv preprint arXiv:2012.12877}, </br> year={2020} </br> } </br> </p></details> </br>
| [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | ICLR | Reg | [GitHub](https://github.com/google-research/vision_transformer) | <details> <summary>Bib</summary> <p align="left">   </br> @article{dosovitskiy2020image, </br> title={An image is worth 16x16 words: Transformers for image recognition at scale}, </br> author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others}, </br> journal={arXiv preprint arXiv:2010.11929}, </br> year={2020} </br> } </p></details> </br>
| [Toward Transformer-Based Object Detection](https://arxiv.org/abs/2012.09958) | ArXiv | Det | --- | <details> <summary>Bib</summary> <p align="left"> </br> @article{beal2020toward, </br> title={Toward Transformer-Based Object Detection}, </br> author={Beal, Josh and Kim, Eric and Tzeng, Eric and Park, Dong Huk and Zhai, Andrew and Kislyuk, Dmitry}, </br> journal={arXiv preprint arXiv:2012.09958}, </br> year={2020} </br> } </p></details> </br>
| [Rethinking Transformer-based Set Prediction for Object Detection](https://arxiv.org/abs/2011.10881) | ArXiv | Det | --- | <details> <summary>Bib</summary> <p align="left">   </br> @article{sun2020rethinking, </br> title={Rethinking Transformer-based Set Prediction for Object Detection}, </br> author={Sun, Zhiqing and Cao, Shengcao and Yang, Yiming and Kitani, Kris}, </br> journal={arXiv preprint arXiv:2011.10881}, </br> year={2020} </br> } </p></details> </br>
| [UP-DETR: Unsupervised Pre-training for Object Detection with Transformers](https://arxiv.org/abs/2011.09094) | ArXiv | Det | --- | <details> <summary>Bib</summary> <p align="left">   </br> @article{dai2020up, </br> title={UP-DETR: Unsupervised Pre-training for Object Detection with Transformers}, </br> author={Dai, Zhigang and Cai, Bolun and Lin, Yugeng and Chen, Junying}, </br> journal={arXiv preprint arXiv:2011.09094}, </br> year={2020} </br> } </p></details> </br>
| [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) | ArXiv | Det | [ GitHub]( https://github.com/fundamentalvision/Deformable-DETR) | <details> <summary>Bib</summary> <p align="left">   </br> @article{zhu2020deformable, </br> title={Deformable DETR: Deformable Transformers for End-to-End Object Detection}, </br> author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng}, </br> journal={arXiv preprint arXiv:2010.04159}, </br> year={2020} </br> } </p></details> </br>
| [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | ECCV | Det | [ GitHub]( https://github.com/facebookresearch/detr) | <details> <summary>Bib</summary> <p align="left">  article{zhu2020deformable, </br>  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection}, </br>  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng}, </br>  journal={arXiv preprint arXiv:2010.04159}, </br>   year={2020} </br> } </br> </p></details>  
| [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840) | Arxiv | Seg | [Github](https://github.com/fudan-zvg/SETR) | <details> <summary>Bib</summary> <p align="left">  @article{zheng2020rethinking, </br>  title={Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers}, </br>   author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jianfeng and Xiang, Tao and Torr, Philip HS and others}, </br>   journal={arXiv preprint arXiv:2012.15840}, </br>   year={2020} </br> }  </br> </p></details>  
| [MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers](https://arxiv.org/abs/2012.00759) | Arxiv | Seg | --- | <details> <summary>Bib</summary> <p align="left">  @article{wang2020max, </br>  title={MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers}, </br>   author={Wang, Huiyu and Zhu, Yukun and Adam, Hartwig and Yuille, Alan and Chen, Liang-Chieh}, </br>   journal={arXiv preprint arXiv:2012.00759}, </br>   year={2020} </br> }  </br> </p></details>  
| [TransTrack: Multiple-Object Tracking with Transformer](https://arxiv.org/abs/2012.15460) | ArXiv | Trk | [GitHub](https://github.com/PeizeSun/TransTrack) | <details> <summary>Bib</summary> <p align="left">   </br> @article{sun2020transtrack, </br> title={TransTrack: Multiple-Object Tracking with Transformer}, </br> author={Sun, Peize and Jiang, Yi and Zhang, Rufeng and Xie, Enze and Cao, Jinkun and Hu, Xinting and Kong, Tao and Yuan, Zehuan and Wang, Changhu and Luo, Ping}, </br> journal={arXiv preprint arXiv:2012.15460}, </br> year={2020} </br> } </p></details> </br>



### 2012-2019

|  Title  |   Venue  |  Task  |   Code   |  BibTeX  |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | NeurIPS'17 | -- | [GitHub](https://github.com/tensorflow/tensor2tensor) | <details> <summary>Bib</summary> <p align="left">  @inproceedings{vaswani2017attention, </br>   title={Attention is all you need}, </br>   author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia}, </br>  booktitle={Advances in neural information processing systems}, </br>  pages={5998--6008}, </br>   year={2017} </br> }  </p></details>

## Awesome vTransformer Libraies
- [WaitingToAdd](https://github.com/penghouwen/VisionTransformer/blob/main/README.md)

