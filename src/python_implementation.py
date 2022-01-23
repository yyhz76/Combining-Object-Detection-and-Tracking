import cv2
import numpy as np
import sys

# Initialize the parameters
objectnessThreshold = 0.5	# Objectness threshold
confThreshold = 0.5			# Confidence threshold
nmsThreshold = 0.4			# Non-maximum suppression threshold (IoU)
inpWidth = 416				# Width of network's input image
inpHeight = 416				# Height of network's input image
classes = []				# Names of classes trained from the coco dataset
MODEL_PATH = './'


def drawPred(classId, conf, left, top, right, bottom, frame):
	#Draw a rectangle displaying the bounding box
	cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3);
	if not classes:
		cv2.CV_Assert(classId < len(classes))

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
		# Scan through all the bounding boxes output from the network and keep only the
		# ones with high confidence scores. Assign the box's class label as the class
		# with the highest score for the box.
        for detection in out:
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                # Get the value and location of the maximum score
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
					# data[0-3] store the position of the bounding box normalized to [0,1] because of the scaling factor 1/255 in blobFromImage()
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append((left, top, width, height))

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    isDetected = False
    box = []
    for i in indices:
        i = i[0]
        if (classes[classIds[i]] == 'sports ball'): 
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)
            isDetected = True
    return isDetected, box





if __name__ == '__main__':
    # Load names of classes
    classesFile = MODEL_PATH + "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = MODEL_PATH + "yolov3.cfg"
    modelWeights = MODEL_PATH + "yolov3.weights"

    # Read network configurations and weights
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    # Use Intel GPU for computation
    net.setPreferableTarget(2)      #cv2.DNN_TARGET_OPENCL_FP16

    # Read video
    cap = cv2.VideoCapture(MODEL_PATH + "soccer-ball.mp4")
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(MODEL_PATH + 'result.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(frame_width), int(frame_height)), 1)

    # Exit if video is not opened
    if not cap.isOpened():
        sys.exit("Could not read video file")

    isTracking = False
    needDetection = False
    isDetected = False
    frameCount = 0
    box = []
    tracker = cv2.TrackerKCF_create()

    while(1):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
        # do detection every 50 frames or when tracking fails
            if frameCount % 50 == 0 or needDetection == True:
                blob = cv2.dnn.blobFromImage(frame,  1/255.0, (inpWidth, inpHeight), (0, 0, 0), True, False)

                #Sets the input to the network
                net.setInput(blob)

                # Runs the forward pass to get output of the output layers
                outs = net.forward(getOutputsNames(net))

                # Remove the bounding boxes with low confidence
                isDetected, box = postprocess(frame, outs)

                # Display detector type on frame
                cv2.putText(frame, "Yolo v3 Detector", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

                # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                t, _ = net.getPerfProfile()
                # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
                # cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(1000.0 / t)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                if isDetected == True:
                    needDetection = False
                else:
                    # Detection failure: output warning message
                    cv2.putText(frame, "Detection failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    needDetection = True
                isTracking = False
            # do tracking for the remaining frames in between
            else:
                if isTracking == False:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, tuple(box))

                # Start timer
                timer = cv2.getTickCount()

                # Update the tracking result
                isTracking, box = tracker.update(frame)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                if isTracking:
                    # Tracking success: Draw the tracked object
                    box = tuple([int(x) for x in box])   
                    cv2.rectangle(frame, box, (0, 255, 0), 2, 1)
                else:
                    # Tracking failure: Display warning message
                    cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    needDetection = True

                # Display tracker type on frame
                cv2.putText(frame, "KCF Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            
            out.write(frame)
            frameCount += 1

            # Display frame.
            cv2.imshow("DetectAndTrack", frame)

            # Exit if ESC pressed.
            k = cv2.waitKey(1)
            if k == 27:
                break
        else:
            break

    cap.release()
    out.release()
