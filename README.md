# combining-detection-and-tracking
Implementation of a fuse object detection + tracking algorithm so that it performs better in terms of both speed and accuracy than a single object tracker.

## TLD tracker
The TLD (Tracking, learning and detection) tracker is robust for objects under occlusion, but produces lots of false positives and is quite slow.

## KCF tracker
KCF (Kernelized Correlation Filter) tracker is fast but does not recover from full occlusion.

## Goal
Combine KCF tracker (fast but not stable under occlusion) with YOLOv3 detector (slow) to make the overall speed and accuracy better than TLD tracker  alone.
