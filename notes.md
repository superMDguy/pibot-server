# Notes on Self Driving Car

## Overview

* Use supervised learning, train to mimic human
* Send last 3-5 frames, and use convnet
* Output: [[steering: right, left, none], [movement: forward, backward, none]]
* FPS: 10, like Nvidia


## Network

* Deep convolutional network, like [nvidia paper](https://arxiv.org/pdf/1604.07316.pdf)
* Use 3D Convolutions over time
* Dense layers at end leading to two outputs

## Training

* Move car around, capture frames, add previous frames, and save.
* Also save csv file with turning: [-1, 0, 1] = [left, none, right]; [-1, 0, 1] = [back, none, forward]
* Use `websockets` to communicate steering between deep learning server and car
* Image dimension: maybe 180x320. Nvidia used 66x200
* Try using YUV?
* Train about 25-50 epochs with `mse` loss
* make sure to keep constant `speed` (170) and `turning_amt` (120).

## Evaluating

* Get last 3 frames, send to network
* Be ready to stop car at all times
* Watch it go :)

