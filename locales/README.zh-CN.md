# 部署 U²-Net 实现智能背景移除

[**English**](../README.md) | **简体中文**

## NVIDIA 系列 AIBOX 

AIBOX-OrinNano 和 AIBOX-OrinNX 均搭载 NVIDIA 原装 Jetson Orin 核心板模组，标配工业级全金属外壳，铝合金结构导热，顶盖外壳侧面采用条幅格栅设计，高效散热，保障在高温运行状态下的运算性能和稳定性，满足各种工业级的应用需求。

| | AIBOX-OrinNX | AIBOX-OrinNano |
| :--- | :--- | :--- |
| 模组 | Jetson Orin NX 16GB | Jetson Orin Nano 8GB |
| AI 性能 | 157 TOPS | 67 TOPS |
| GPU | 搭载 32 个 Tensor Core 的 1024 核 NVIDIA Ampere 架构 GPU | 搭载 32 个 Tensor Core 的 1024 核 NVIDIA Ampere 架构 GPU |
| CPU | 8 核 Arm Cortex - A78 64 位 CPU<br>2MB L2 + 4MB L3 | 6 核 Arm Cortex A78 64 位 CPU<br>1.5MB L2 + 4MB L3 |
| DDR | 16GB 128 位 LPDDR5 102.4GB/s | 8GB 128 位 LPDDR5 68 GB/s |
| HDMI | 4K@60Hz | 4K@30Hz |



## 背景移除

Background Removal（背景移除）技术已成为图像处理领域的重要工具，主要应用于图像编辑、数据分析和应用开发等场景。

典型应用：

（1）**​图像处理‌：​**电商产品图抠像、人像美化、医学影像分析

（2）**​视频处理‌：​**实时绿幕替代、动态物体追踪

‌（3）**​科研预处理‌：​**气象色谱分析中通过背景移除提升量化精度

## U²-Net

U²-Net（U-squared Net）是一种基于深度学习的图像分割模型，专为高精度背景移除任务设计，其核心技术特点和应用场景如下：

- **双U型编解码结构**
- **深监督与损失函数**
- **轻量化设计**

![8](../res/2.webp)


### **下载源码**

```
$ git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
```

**编译 / 安装**

> 参考：[https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md)

**运行示例**

```
# remove the background (with alpha)
$ ./backgroundnet.py images/bird_0.jpg images/test/bird_mask.png                   


# replace the background
$ ./backgroundnet.py --replace=images/snow.jpg images/bird_0.jpg images/test/bird_replace.jpg
```

![8](../res/3.webp)
