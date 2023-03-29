# A Unified Pyramid Recurrent Network for Video Frame Interpolation


<div align="center">
  <img src=figures/pipeline.png width=700 />
</div>


This project is the official implementation of our CVPR 2023 paper, [A Unified Pyramid
Recurrent Network for Video Frame Interpolation](https://arxiv.org/abs/2211.03456).


## Introduction

We present UPR-Net, a novel **U**nified **P**yramid **R**ecurrent Network for
frame interpolation. Cast in a flexible pyramid framework, UPR-Net exploits
lightweight recurrent modules for both bi-directional flow estimation and
intermediate frame synthesis. At each pyramid level, it leverages estimated
bi-directional flow to generate forward-warped representations for frame
synthesis; across pyramid levels, it enables iterative refinement for both
optical flow and intermediate frame. In particular, we show that our iterative
synthesis strategy can significantly improve the robustness of frame
interpolation on large motion cases. Despite being extremely lightweight (1.7M
parameters), our base version of UPR-Net achieves excellent performance on a
large range of benchmarks.


<p float="center">
  <img src=figures/accuracy-efficiency-snufilm.png width=400 />
  <img src=figures/accuracy-efficiency-vimeo90k.png width=400 />
</p>


<div align="center">
  <img src=figures/snufilm-visualization.png width=1000 />
</div>


## Python and Cuda environment
This code has been tested with PyTorch 1.6 and Cuda 10.2. It should also be
compatible with higher versions of PyTorch and Cuda. Run the following command
to initialize the environment:

```
conda create --name uprnet python=3.7
conda activate uprnet
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip3 install cupy_cuda102==9.4.0
pip3 install -r requirements.txt
```

In particular, CuPy package is required for running the forward warping
operation (refer to
[softmax-splatting](https://github.com/sniklaus/softmax-splatting) for details).
If your Cuda version is lower than 10.2 (not lower than 9.2), we suggest to
replace `cudatoolkit=10.2` in above command with `cudatoolkit=9.2`, and replace
`cupy_cuda102==9.4.0` with `cupy_cuda92==9.6.0`.


## Play with demo
We place trained model weights in `checkpoints`, and provide a script to test
our frame interpolation model. Given two consecutive input frames, and the
desired time step, run the following command, then you will obtain estimated
bi-directional flow and interpolated frame in the `./demo/output` directory.
```
python3 -m demo.interp_imgs \
--frame0 demo/images/beanbags0.png \
--frame1 demo/images/beanbags1.png \
--time_period 0.5
```
Here the `time_period` (float number in 0~1) indicates the time step of the
intermediate frame you want to interpolate.


## Training on Vimeo90K
By default, our model is trained on Vimeo90K. If you want to train our model,
please download [Vimeo90K](http://toflow.csail.mit.edu/).

### Default training configuration
You can run the following command to train the base version of our UPR-Net:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=10000 -m tools.train \
        --world_size=4 \
        --data_root /path/to/vimeo_triplet \
        --train_log_root /path/to/train_log \
        --exp_name upr-base \
        --batch_size 8 \
        --nr_data_worker 2
```

Please (must) assign `data_root` with the path of vimeo_triplet for training, and
(optionally) assign `train_log_root`  with the path to save logs (trained weights and
tensorboard logs).  We do not recommend saving log files under dir of this
codebase.  If `train_log_root` is not explicitly assigned, all logs will be
saved in `../upr-train-log` by default. We also intentionally soft link
`train_log_root` to `./train-log` for convenience.


### Some tips for training
- If you want to train large or LARGE versions of our UPR-Net, please assign the
    argument `model_size` as `large` or `LARGE`.

- If you have suspended the training, and want to restart the training from 
    previous checkpoint, please assign the argument `resume` as `True` in the
    training command.

- By default, we set total batch_size as 32, and use 4 GPUs for distributed
    training, with each GPU processing 8 samples in a batch (`batch_size` is set
    as 8 in our training command). Therefore, if you use 2 GPUs for training, please set
    `batch_size` as 16 in the training command.

- You can view the training curve, interpolation, and optical flow using
    TensorBoard, by running command like `tensorboard
    --logdir=./train-log/upr-base/tensorboard`.


## Benchmarking

#### Trained model weights
We have placed our trained model weights in `./checkpoints`. The weights of
base/large/LARGE versions of our UPR-Net are named as `upr.pkl`,
`upr_large.pkl`, `upr_llarge.pkl`, respectively.


#### Benchmark datasets
We evaluate our UPR-Net series on Vimeo90K, UCF101, SNU-FILM, and 4K1000FPS.

If you want to train and benchmark our model, please download
[Vimeo90K](http://toflow.csail.mit.edu/),
[UCF101](https://liuziwei7.github.io/projects/VoxelFlow),
[SNU-FILM](https://myungsub.github.io/CAIN/),
[4K1000FPS](https://github.com/JihyongOh/XVFI#X4K1000FPS).


#### Benchmarking scripts
We provide scripts to test frame interpolation accuracy on Vimeo90K, UCF101,
SNU-FILM, and 4K1000FPS. You should configure the path to benchmark datasets
when running these scripts.

```
python3 -m tools.benchmark_vimeo90k --data_root /path/to/vimeo_triplet/
python3 -m tools.benchmark_ucf101 --data_root /path/to/ucf101/
python3 -m tools.benchmark_snufilm --data_root /path/to/SNU-FILM/
python3 -m tools.benchmark_8x_4k1000fps --test_data_path /path/to/4k1000fps/test
```
By default, we test the base version of UPR-Net. To test the large/LARGE
versions, please change corresponding arguments (`model_size` and `model_file`)
in benchmarking scripts.


Additionally, run the following command can test our runtime.
```
python -m tools.runtime
```

#### Our benchmarking results
Our benchmarking results on UCF101, Vimeo90K, SNU-FILM are shown in below table.
You can verify our results by running our benchmarking scripts. Runtime is
measured with a single 2080TI GPU for interpolating two 640x480 frames.

<div align="center">
  <img src=figures/quantitative.png width=1000 />
</div>

Our benchmarking results on 4K1000FPS are shown in below table.

<div align="left">
  <img src=figures/quantitative-4k.png width=400 />
</div>


## Acknowledgement
We borrow some codes from
[RIFE](https://github.com/megvii-research/ECCV2022-RIFE),
[softmax-splatting](https://github.com/sniklaus/softmax-splatting), and
[EBME](https://github.com/srcn-ivl/EBME). We thank the
authors for their excellent work. When using our code, please also pay attention
to the licenses of RIFE, softmax-splatting, and EBME.


## Citation
```
@inproceedings{jin2023unified,
  title={A Unified Pyramid Recurrent Network for Video Frame Interpolation},
  author={Jin, Xin and Wu, Longhai and Chen, Jie and Chen, Youxin and Koo,
  Jayoon and Hahm, Cheul-hee},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern
  recognition},
  year={2023}
}
```
