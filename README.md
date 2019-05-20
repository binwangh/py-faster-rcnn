## VGG网络特性
- 整个网络使用卷积核尺寸（3x3）和最大池化尺寸（2x2）
- 用ReLU激活函数
- VGG(*)：卷积层和全连接层参与计数，池化层不参与计数 

## ZF网络特性

## VGG_CNN_M_1024-Feature Map模块 [K,S,P]
- Convolution/ReLU/LRN/Pooling [7,2,0]  [3,2,0]
- Convolution/ReLU/LRN/Pooling [5,2,1]  [3,2,0]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]


## VGG16-Feature Map模块 [K,S,P]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU/Pooling     [3,1,1]  [2,2,0]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU/Pooling     [3,1,1]  [2,2,0]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU/Pooling     [3,1,1]  [2,2,0]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU/Pooling     [3,1,1]  [2,2,0]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]


## ZF-Feature Map模块 [K,S,P]
- Convolution/ReLU/LRN/Pooling [7,2,3]  [3,2,1]
- Convolution/ReLU/LRN/Pooling [5,2,2]  [3,2,1]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]
- Convolution/ReLU             [3,1,1]

## RPN模块 [K,S,P]
- Convolution/ReLU                                                [3,1,1]
	- Convolution 【rpn_cls_score】 + Reshape + Softmax + Reshape   [1,1,0]
	- Convolution 【rpn_bbox_pred】                                 [1,1,0]
- Python
- Python

# 目录说明
- caffe-fast-rcnn
- data
- experiments
- lib
- models
	- coco 【类别数：84】
		- VGG_CNN_M_1024
			- fast_rcnn
				- solver.prototxt
				- test.prototxt
				- train.prototxt：SoftmaxWithLoss、SmoothL1Loss两个
				- fast_rcnn网络
					- VGG_CNN_M_1024-Feature Map模块
					- ROIPooling
					- InnerProduct/ReLU 
					- InnerProduct/ReLU
					- InnerProduct 【cls_score】
					- InnerProduct 【bbox_pred】
			- faster_rcnn_end2end：num_output: 24   # 2(bg/fg) * 12(anchors)
				- solver.prototxt
				- test.prototxt
				- train.prototxt：SoftmaxWithLoss、SmoothL1Loss、SoftmaxWithLoss、SmoothL1Loss四个
				- faster_rcnn_end2end网络
					- VGG_CNN_M_1024-Feature Map模块
					- RPN模块
					- ROIPooling
					- InnerProduct/ReLU
					- InnerProduct/ReLU
					- InnerProduct 【cls_score】
					- InnerProduct 【bbox_pred】
		- VGG16
			- fast_rcnn
				- solver.prototxt
				- test.prototxt
				- train.prototxt
				- fast_rcnn网络
					- VGG16-Feature Map模块
					- ROIPooling
					- InnerProduct/ReLU 
					- InnerProduct/ReLU
					- InnerProduct 【cls_score】
					- InnerProduct 【bbox_pred】
			- faster_rcnn_end2end：num_output: 24   # 2(bg/fg) * 12(anchors)
				- solver.prototxt
				- test.prototxt
				- train.prototxt
				- faster_rcnn_end2end网络
					- VGG16-Feature Map模块
					- RPN模块
					- ROIPooling
					- InnerProduct/ReLU
					- InnerProduct/ReLU
					- InnerProduct 【cls_score】
					- InnerProduct 【bbox_pred】
	- pascal_voc 【类别数：21】
		- VGG_CNN_M_1024
			- fast_rcnn：相比于coco下的：在全连接层之前多了Dropout层
			    - solver.prototxt
				- test.prototxt
				- train.prototxt
			- faster_rcnn_alt_opt(!!!!!!!!!!)
				- faster_rcnn_test.pt
				- rpn_test.pt 【2】【5】
				- stage1_fast_rcnn_solver30k40k.pt
				- stage1_fast_rcnn_train.pt 【3】
				- stage1_rpn_solver60k80k.pt
				- stage1_rpn_train.pt 【1】
				- stage2_fast_rcnn_solver30k40k.pt
				- stage2_fast_rcnn_train.pt 【6】
				- stage2_rpn_solver60k80k.pt
				- stage2_rpn_train.pt 【4】
			- faster_rcnn_end2end：num_output: 18   # 2(bg/fg) * 9(anchors)
				- solver.prototxt
				- test.prototxt
				- train.prototxt
		- VGG16
			- fast_rcnn
				- solver.prototxt
				- test.prototxt
				- train.prototxt
			- faster_rcnn_alt_opt(!!!!!!!!!)
				- faster_rcnn_test.pt
				- rpn_test.pt 【2】【5】
				- stage1_fast_rcnn_solver30k40k.pt
				- stage1_fast_rcnn_train.pt 【3】
				- stage1_rpn_solver60k80k.pt
				- stage1_rpn_train.pt 【1】
				- stage2_fast_rcnn_solver30k40k.pt
				- stage2_fast_rcnn_train.pt 【6】
				- stage2_rpn_solver60k80k.pt
				- stage2_rpn_train.pt 【4】
			- faster_rcnn_end2end
				- solver.prototxt
				- test.prototxt
				- train.prototxt
		- ZF
			- fast_rcnn
				- solver.prototxt
				- test.prototxt
				- train.prototxt
				- fast_rcnn网络
					- ZF-Feature Map模块
					- ROIPooling
					- InnerProduct/ReLU/Dropout 
					- InnerProduct/ReLU/Dropout
					- InnerProduct 【cls_score】
					- InnerProduct 【bbox_pred】
			- faster_rcnn_alt_opt(!!!!!!!!!)
				- faster_rcnn_test.pt
				- rpn_test.pt 【2】【5】
				- stage1_fast_rcnn_solver30k40k.pt
				- stage1_fast_rcnn_train.pt 【3】
				- stage1_rpn_solver60k80k.pt
				- stage1_rpn_train.pt 【1】
				- stage2_fast_rcnn_solver30k40k.pt
				- stage2_fast_rcnn_train.pt 【6】
				- stage2_rpn_solver60k80k.pt
				- stage2_rpn_train.pt 【4】
			- faster_rcnn_end2end
				- solver.prototxt
				- test.prototxt
				- train.prototxt
- tools
- README.md

# py-faster-rcnn has been deprecated. Please see [Detectron](https://github.com/facebookresearch/Detectron), which includes an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870).
# py-faster-rcnn已经被弃用（尴尬，但是可以从中学习一些检测的思想！！！）  可以看[facebookresearch/Detectron：包含Mask R-CNN的实现](https://github.com/facebookresearch/Detectron)

## [faster-rcnn知乎讲解，相当全面](https://zhuanlan.zhihu.com/p/31426458)

### Disclaimer（免责声明）

[官方Faster R-CNN代码由MATLAB实现](https://github.com/ShaoqingRen/faster_rcnn)，如果想复现NIPS 2015 paper，需要用到此代码。

仓库包含了Python代码：MATLAB版本的重新实现。
Python的实现Fork来源于[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn).
但是这个两个版本在实现上会有细微的差别。
尤其，Python端口
- 在测试阶段，慢了10%，because some operations execute on the CPU in Python layers。（利用VGG16：220ms / image vs. 200ms / image）
- 给出了与MATLAB相似，但不完全相同的：mAP [mean Average Precision] ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
- 与使用MATLAB代码训练的模型不兼容，由于实现上有细微的差别
- **includes approximate joint training** that is 1.5x faster than **alternating optimization** (for VGG16)
	- [详细细节文档：slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0)
	- 已下载：doc文件夹下

# 论文：Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- 作者：Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)
- This Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.
- 更多细节请看官方[MATLAB版本：README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md)
- [论文地址](http://arxiv.org/abs/1506.01497),并于2015年在NIPS上发表。

### License（许可证）

Faster R-CNN is released under the MIT License（麻省理工学院） (refer to the LICENSE file for details).

### Citing Faster R-CNN（引用）

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software（所需软件）

**注意** 如果在编译时遇到问题，并且使用的是CUDA/CUDNN的最新版本，可以参考[这个链接：获取解决方案](https://github.com/rbgirshick/py-faster-rcnn/issues/509?_pjax=%23js-repo-pjax-container#issuecomment-284133868) 

1. 需要 'Caffe' 和 'pycaffe'，参考[Caffe 安装介绍](http://caffe.berkeleyvision.org/installation.html)

  **注意：** Caffe在编译时，必须支持Python层，即打开 WITH_PYTHON_LAYER 开关，如下所示。

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
 
  可以下载，并参考[作者的 Makefile.config](https://dl.dropboxusercontent.com/s/6joa55k64xo2h68/Makefile.config?dl=0)

2. 可能没有的Python安装包：`cython`, `python-opencv`, `easydict`
- 需要单独安装，还是？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

3. [可选] MATLAB接口被需要，仅仅在 PASCAL VOC 评价时被用到。 可以选用非官方的Python代码来进行评价。（代码已经集成）

### Requirements: hardware

1. 对于小网络的训练（ZF，VGG_CNN_M_1024），需要一个好的GPU（Titan，K20，K40，...）至少要求3GB的显存
2. 对于训练Fast R-CNN-VGG16，需要K40,11GB显存
3. 对于训练端到端版本的Faster R-CNN-VGG16，在CUDNN的情况下，需要3GB的显存 【CUDNN可以减少显存的使用量？？？？？？？？？？？？？？？？？？？？？？】


### Basic Installation (sufficient（足够的） for the demo)

1. 克隆Faster R-CNN repository（仓库）**【强烈建议用下面的命令进行操作】**
  ```Shell
  # Make sure to clone with --recursive（递归）
  git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
  ```

2. 我们会将Faster R-CNN克隆至根目录 `FRCN_ROOT`

   *如果按照上面的步骤1进行操作，则可以忽略注释1和注释2*
   *Ignore notes 1 and 2 if you followed step 1 above.*

    **注释 1：** 如果你没有使用标识 '--recursive'，你需要手动克隆子模块 'caffe-fast-rcnn'
    ```Shell
    git submodule update --init --recursive
    ```
	**注释 2：** 子模块 'caffe-fast-rcnn' 需要位于 'faster-rcnn' 分支下（等效的分离状态）。如果按照步骤1的介绍来操作，这个会自动运行。

3. Build the Cython modules（编译Cython模块）
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe（编译Caffe和pycaffe）
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download pre-computed Faster R-CNN detectors（下载预先计算的Faster R-CNN 检测器）
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_faster_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
    These models were trained on VOC 2007 trainval.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data/README.md`
7. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
