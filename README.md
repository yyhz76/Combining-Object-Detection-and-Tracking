# Combining-Detection-and-Tracking
Implementing a fused object detection + tracking algorithm so that it performs better in terms of both speed and accuracy than a single object tracker.

## TLD tracker
The TLD (Tracking, Learning and Detection) tracker is robust for objects under occlusion, but produces lots of false positives and is quite slow.

![alt text](https://github.com/yyhz76/ORB-Feature-based-Image-Alignment/blob/main/images/BGR_channels_in_grayscale.png)<br /><br />  

## KCF tracker
KCF (Kernelized Correlation Filter) tracker is fast but does not recover from full occlusion.

![alt text](https://github.com/yyhz76/ORB-Feature-based-Image-Alignment/blob/main/images/BGR_channels_in_grayscale.png)<br /><br /> 

## Goal
Combine KCF tracker (fast but not stable under occlusion) with YOLOv3 detector (slow) to make the overall speed and accuracy better than TLD tracker  alone.
