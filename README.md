# Combining-Detection-and-Tracking
Implementing a fused object detection + tracking algorithm so that it performs better in terms of both speed and accuracy than a single object tracker.

## TLD tracker
The TLD (Tracking, Learning and Detection) tracker is robust for objects under occlusion, but produces lots of false positives and is quite slow.

![alt text](https://github.com/yyhz76/combining-detection-and-tracking/blob/main/images/TLD_alone.png)<br /><br />  

## KCF tracker
KCF (Kernelized Correlation Filter) tracker is fast but does not recover from full occlusion.

![alt text](https://github.com/yyhz76/combining-detection-and-tracking/blob/main/images/KCF_alone.png)<br /><br /> 

## Goal
Combine KCF tracker (fast but not stable under occlusion) with YOLOv3 detector (slow) to make the overall speed and accuracy better than TLD tracker alone.



## Idea
* Specify an initial bounding box of the soccer ball for the YOLOv3 detector
* Perform detection every 50 frames OR when the tracker fails
* Perform tracking for in-between frames

## Video Demo
Click here for the video demo


* Blue bounding box:  YOLOv3 detector in use
* Green bounding box: KCF tracker in use
* The display indicates a warning message when detection/tracking fails

## Observations
The video demo link above combines the following 3 cases together:
  * Top: KCF tracker alone: 		Highest FPS (frame per second), but the tracker fails after encountering occlusion
  * Center: TLD tracker alone:		Accuracy is better than KCF alone, but is not stable and produces lots of false positives. Performance is slow (Low FPS). 
  * Bottom: KCF tracker + YOLOv3 detector:	Accuracy is much better than TLD alone, as the detector can retrieve the correct location of the ball if KCF tracker fails. Performance is much faster than TLD alone since most of the time KCF tracker does the work. This is a combination of the advantages from both KCF tracker (performance) and YOLOv3 detector (accuracy).	

![alt text](https://github.com/yyhz76/combining-detection-and-tracking/blob/main/images/combined.png)<br /><br /> 

