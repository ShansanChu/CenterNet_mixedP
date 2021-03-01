# CenterNet mixedPrecision inference
This repo is heavily borrowed from [CenterNet](https://github.com/xingyizhou/CenterNet). We ad FP16 inference of Hourglass human pose detection. We are working on INT8.

##FP16 inference precision.
Hourglass 1x accuracy is from 58 mAP to 54 mAP. And for the Hourglass 3x accuracy is from 64 mAP to 61.2 mAP. Torch.clamp is used for fp16 scale=1 saturation quantization.  
