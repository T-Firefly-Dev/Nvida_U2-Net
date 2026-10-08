# Deploying U²-Net for Intelligent Background Removal

## Background Removal

Background removal is essentially a pixel-level binary classification problem: every pixel must be classified as foreground or background. The technology has become an essential tool in image processing and acts as a pre-processing stage in a number of business pipelines:

- **E-commerce and content production**: product cutouts, batch background replacement, and portrait retouching. The per-call cost of a cloud API may appear modest, but once daily volume reaches the tens of thousands, call cost, upload bandwidth, and asset privacy all become constraining factors.
- **Video conferencing and live streaming**: real-time background blurring/replacement as an alternative to green screens. These scenarios are highly latency-sensitive — a single cloud round trip typically costs tens to hundreds of milliseconds — which makes on-device inference the practical path to meeting real-time requirements.
- **Machine vision pre-processing**: suppressing the background before feeding frames to detection/classification networks is effectively an attention crop. It measurably improves downstream accuracy in cluttered scenes while reducing the compute spent on irrelevant pixels.

Deploying these workloads on the AIBOX at the edge yields three core benefits: **image data never leaves the device** (privacy- and compliance-friendly), **no per-call cost** (deploy once, use indefinitely), and **offline availability** (stable operation in factories, stores, and other bandwidth-constrained environments).

## NVIDIA AIBOX Series

Both the AIBOX-OrinNano and AIBOX-OrinNX are equipped with original NVIDIA Jetson Orin core modules. They come standard with an industrial-grade all-metal enclosure featuring an aluminum alloy structure for thermal conduction. The top cover utilizes a slatted grille design on the sides for highly efficient heat dissipation, ensuring computational performance and stability under high-temperature operation to meet the demands of various industrial applications.

| | AIBOX-OrinNX | AIBOX-OrinNano |
| :--- | :--- | :--- |
| Module | Jetson Orin NX 16GB | Jetson Orin Nano 8GB |
| AI Performance | 157 TOPS | 67 TOPS |
| GPU | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores |
| CPU | 8-core Arm Cortex-A78 64-bit CPU<br>2MB L2 + 4MB L3 | 6-core Arm Cortex-A78 64-bit CPU<br>1.5MB L2 + 4MB L3 |
| DDR | 16GB 128-bit LPDDR5 102.4GB/s | 8GB 128-bit LPDDR5 68GB/s |
| HDMI | 4K@60Hz | 4K@30Hz |

## U²-Net Model Analysis

U²-Net was proposed by Qin Xuebin et al. of the University of Alberta (Pattern Recognition 2020, 2020 Best Paper Award; arXiv:2005.09007). Its target problem is salient object detection (SOD): without a specified class, the network automatically localizes the most visually salient subject region in the frame — precisely the objective of background removal.

![U²-Net network architecture](res/1.webp)

### Nested U-shaped Structure (RSU Blocks)

A standard U-Net adopts a single-level encoder–decoder structure. U²-Net instead constructs each stage as a small U-Net of its own (RSU, ReSidual U-block), forming a **two-level nested U-shaped architecture** — hence the "U²" in its name.

The mechanism can be compared to **nested interrupt handling** in embedded systems: a standard U-Net is equivalent to a single level of interrupt nesting, whereas an RSU allows an interrupt service routine to be nested within itself. Each stage independently performs "downsampling to capture global context, upsampling to restore local detail" at its own resolution. As a result, the network acquires multi-scale contextual information and local high-frequency detail simultaneously, without a significant increase in computational cost — critical for segmentation quality on hair, thin structures, and object boundaries.

### Deep Supervision

Each decoder level outputs its own saliency prediction map; the losses are computed separately and then fused with weighting to produce the final result. This resembles **placing a checkpoint at every stage of a pipeline** rather than inspecting only the final output: gradient signals propagate back to shallow layers more effectively, which keeps training stable even for relatively small models.

### Specifications and Training Data

| Item | Value |
| --- | --- |
| Input resolution used for reported metrics | 320×320 |
| Standard weights (u2net.pth) | 176.3 MB |
| Lightweight weights (u2netp.pth) | 4.7 MB |
| Training set | DUTS-TR (10,553 annotated saliency images) |
| Official variant | u2net_human_seg (portrait segmentation, trained on the Supervisely Person Dataset) |

The `backgroundNet` implementation in jetson-inference is based on a **fully convolutional U²-Net**, and therefore supports arbitrary input resolutions: with no fully connected layers in the network, the input size is not limited to 320×320. Higher input resolutions yield finer boundary detail at the cost of longer inference time, so deployments need to trade off accuracy against frame rate.

## Deployment Walkthrough

### System Preparation

Maximum performance mode:

```bash
$ sudo nvpmodel -m 0        # MAX-N mode (Orin Nano defaults to 15W)
$ sudo jetson_clocks        # lock CPU/GPU/EMC clocks to maximum
```

> Prerequisite: the AIBOX must be flashed with JetPack 5.x/6.x — the versions officially supported by jetson-inference.

### Download the Source Code and Build

```bash
$ git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
```

Refer to the official documentation for the build and installation procedure:
[https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md)

> Alternatively, use the Docker workflow bundled with the project: `cd jetson-inference && docker/run.sh`. All samples are pre-built inside the image, and the executables are located in the container's `build/aarch64/bin/` directory.

### Static Images: Background Removal / Replacement

```bash
# remove the background (with alpha)
$ ./backgroundnet.py images/bird_0.jpg images/test/bird_mask.png

# replace the background with a given image (auto-rescaled to input resolution)
$ ./backgroundnet.py --replace=images/snow.jpg images/bird_0.jpg images/test/bird_replace.jpg
```

**First-run note**: on first execution the program automatically downloads the `Background-U2Net/u2net.onnx` model, after which TensorRT compiles an optimized engine for the current device's GPU configuration. This step takes **approximately 10 minutes** (figure from public benchmarks). The compiled artifact is cached, so subsequent launches run at normal speed.

![Background removal and replacement result](res/2.webp)

### Live Video Streams: Camera / Video File / RTSP

`backgroundnet` follows the unified stream protocol of jetson-inference for both input and output, so switching the input source requires no code changes:

```bash
# live background removal from a CSI or USB camera
$ ./backgroundnet.py /dev/video0

# live background replacement from a camera
$ ./backgroundnet.py --replace=images/coral.jpg /dev/video0

# video file (full URI required)
$ ./backgroundnet.py file:///absolute/path/to/video.mp4

# network camera / surveillance RTSP stream
$ ./backgroundnet.py rtsp://192.168.1.100:554/stream
```

The output side is equally flexible: by default frames are rendered to a display, but they can also be written to a video file via an output stream, or streamed over WebRTC for viewing in a browser on the local network — a good fit for the typical deployment pattern where the device sits on the production floor while operators monitor from an office area.
