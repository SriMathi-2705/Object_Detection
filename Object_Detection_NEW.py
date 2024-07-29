import cv2
import numpy as np

# Load YOLO
print("LOADING YOLO")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Save all the names in the file to the list classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layers of the network
layer_names = net.getLayerNames()

# Determine the output layer names from the YOLO model
output_layers = [i for i in range(len(layer_names)) if layer_names[i].endswith("yolo_")]

print("YOLO LOADED")

# Open a connection to the camera
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, change it if you have multiple cameras

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Failed to open camera.")
    exit()

while True:
    # Read and preprocess the input frame from the camera
    ret, frame = cap.read()
    
    # Check if frame retrieval was successful
    if not ret:
        print("Error: Unable to retrieve frame from the camera.")
        break
    
    height, width, channels = frame.shape

    # Using the blob function of OpenCV to preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Use NMS function in OpenCV to perform Non-maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            # Display object name and confidence
            text = f"{label}: {confidence:.2f}"
            # Display the object name and confidence above the rectangle
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw the rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the frame with detected objects
    cv2.imshow("Live Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
