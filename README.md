# Real-time Human Fighting Detection

![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains code for real-time human fighting detection using a pre-trained deep learning model. The code includes components for video input from cameras or video files, real-time inference, and the display of classification results.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Customization](#customization)
- [Results](#results)
- [Contributing](#contributing)

## Prerequisites

Before running the code, ensure you have the following dependencies installed. You can install these packages using the provided `requirements.txt` file.


## Getting Started
```bash
git clone https://github.com/tirohan/real-time-fighting-detection.git
```

```bash
cd real-time-fighting-detection
```

##Usage
To run the code with a USB camera, use the following command:
```bash
python cam_gui.py
```
This will start real-time inference using your default camera.

To run the code with a video file, use the following command:
```bash
python video_gui.py
```
## Switching Cameras
While the code is running, you can switch between available cameras (if more than one is connected) by clicking the "Switch Camera" button in the GUI.

## Quitting Inference
To quit the inference process, click the "Quit Inference" button in the GUI.

## Results
The code will display real-time video with overlaid classification results, including class labels, confidence scores, timestamps (for violence detections), and inference times.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

