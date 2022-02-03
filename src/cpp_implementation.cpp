// A combination of object detection and tracking to improve overall performace and accuracy
// Detector: YOLO_v3
// Tracker:  KCF tracker

// Run detection every 50 frames OR detection/tracking failure occurs
// Run tracking on in-between frames

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <typeinfo>
#include <fstream>

using namespace std;
using namespace cv;

// Input files
const string inputVideoPath = "soccer-ball.mp4";
const string outputVideoPath = "result.mp4";
const string fileClassNames = "coco.names";		// class names of coco dataset
const string modelConfiguration = "yolov3.cfg";
const string modelWeights = "yolov3.weights";

// Initialize the parameters
float objectnessThreshold = 0.5;			// Objectness threshold
float confThreshold = 0.5;				// Confidence threshold
float nmsThreshold = 0.4;				// Non-maximum suppression threshold (IoU)
int inWidth = 416;					// Width of network's input image
int inHeight = 416;					// Height of network's input image

void drawPred(const vector<string>& classNames, int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 0, 0), 3);
	if (!classNames.empty())
	{
		CV_Assert(classId < (int)classNames.size());
	}
}

// Remove the bounding boxes with low confidence using nonmax suppression
bool postprocess(Mat& frame, const vector<string>& classNames, const vector<Mat>& outs, Rect& box)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			if (outs[i].at<float>(j, 4) > objectnessThreshold) {				// if there is an object in the bounding box
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold)						// if a certain class is considered detected
				{
					// data[0-3] store the position of the bounding box normalized to [0,1] because of the scaling factor 1/255 in blobFromImage()
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}
	}

	// Perform nonmax suppression to eliminate redundant overlapping boxes with lower confidences
	// Only box indices in 'indices' are retained as true bounding boxes
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	bool isDetected = false;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		if (classNames[classIds[idx]] == "sports ball") {
			box = boxes[idx];
			drawPred(classNames, classIds[idx], confidences[idx], box.x, box.y,
				box.x + box.width, box.y + box.height, frame);
			isDetected = true;
		}
	}
	return isDetected;
}

// Do detection
void detect(Mat& frame, dnn::Net& net, vector<string>& classNames, Rect& box, bool& needDetection, bool& hasTracker) {
	Mat blob;
	dnn::blobFromImage(frame, blob, 1 / 255.0, Size(inWidth, inHeight), Scalar(0, 0, 0), true, false);

	// Feed the input to the network
	net.setInput(blob);

	// Run forward pass to get bounding boxes of the output layers (3 output layers at scale 13 * 13, 26 * 26, 52 * 52, each grid cell outputs 3 bounding boxes)
	// 'outs' is an array of 3 Mat. 
	// outs[0] contains 13 * 13 * 3 = 507 rows, each row is a bounding box, containing 85 elements (centerx, centery, width, height, confidence of objectness, confidence of 80 classes)
	// outs[1] contains 26 * 26 * 3 = 2028 rows
	// outs[2] contains 52 * 52 * 3 = 8112 rows
	vector<Mat> outs;
	net.forward(outs, net.getUnconnectedOutLayersNames());	// get the output bounding boxes from the 3 output layers and save them in 'outs'

	// Remove redundant bounding boxes, the box location detecting the soccer box is saved in 'box' and is used for KCF tracker
	bool isDetected = postprocess(frame, classNames, outs, box);

	// Display detector type on frame
	cv::putText(frame, "Yolo v3 Detector", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;

	// Display FPS on frame
	cv::putText(frame, "FPS : " + to_string(int(1000.0 / t)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

	if (isDetected) {
		needDetection = false;
	}
	else {
		// Detection failure: output warning message
		cv::putText(frame, "Detection failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		needDetection = true;
	}

	hasTracker = false;
}

// Do tracking
void track(Mat& frame, Ptr<Tracker>& tracker, Rect& box, bool& needDetection, bool& hasTracker) {
	if (!hasTracker) {
		tracker = TrackerKCF::create();
		tracker->init(frame, box);
		hasTracker = true;
	}

	// Start timer
	double timer = (double)getTickCount();

	// Update the bounding box location and save into 'box'
	bool isTracking = tracker->update(frame, box);

	// Calculate Frames per second (FPS)
	float fps = getTickFrequency() / ((double)getTickCount() - timer);

	if (isTracking)
	{
		// Tracking success: Draw the tracked object
		rectangle(frame, box, Scalar(0, 255, 0), 2, 1);
	}
	else
	{
		// Tracking failure: Display warning message. Set flag for detection
		cv::putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		needDetection = true;
	}

	// Display tracker type on frame
	cv::putText(frame, "KCF Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

	// Display FPS on frame
	cv::putText(frame, "FPS : " + to_string(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
}

int main() {
	// Load names of classes
	vector<string> classNames;
	ifstream ifs(fileClassNames.c_str());
	string line;
	while (getline(ifs, line)) classNames.push_back(line);

	// Load the network
	dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);

	// Use Intel GPU for computation
	net.setPreferableTarget(dnn::DNN_TARGET_OPENCL_FP16);

	// Read input video
	VideoCapture video(inputVideoPath);
	int frame_width = video.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = video.get(CAP_PROP_FRAME_HEIGHT);
	double fps = video.get(CAP_PROP_FPS);

	if (!video.isOpened())
	{
		std::cout << "Could not read video file" << endl;
		return 1;
	}

	// Configure output video
	VideoWriter out(outputVideoPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frame_width, frame_height));

	int frameCount = 0;
	Mat frame;
	Rect box;
	Ptr<Tracker> tracker;

	bool needDetection = false;
	bool hasTracker = false;			// ensures that the same tracker is reused if previous tracking is successful

	while (video.read(frame))
	{
		// Do DETECTION every 50 frames or when detection / tracking fails
		if (frameCount % 50 == 0 || needDetection) 
			detect(frame, net, classNames, box, needDetection, hasTracker);

		// Do TRACKING for the remaining frames in between
		else
			track(frame, tracker, box, needDetection, hasTracker);

		out.write(frame);
		frameCount++;

		// Display frame
		cv::imshow("DetectAndTrack", frame);

		// Exit if ESC pressed
		int k = waitKey(1);
		if (k == 27) break;
	}

	video.release();
	out.release();

	return 0;
}
