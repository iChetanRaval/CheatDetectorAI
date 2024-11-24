import cv2
import dlib
import numpy as np
import datetime

# Load the YOLO object detection model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load YOLO object labels (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load pre-trained face detector and shape predictor models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_landmarks(landmarks, eye_indices):
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices], dtype="int")


# Left and Right Eye indices
left_eye_indices = [36, 37, 38, 39, 40, 41]
right_eye_indices = [42, 43, 44, 45, 46, 47]

# Open the default webcam
cap = cv2.VideoCapture(0)

# Variables for tracking eye contact percentage
eye_contact_count = 0
total_frames = 0
eye_contact_threshold = 0.25  # Lowered threshold to make eye contact detection more responsive

# Scale factor for increasing the detection area
scale_factor = 0.9  # Reduced to 90% for better focus and smaller window

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for a smaller detection area
    frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    height, width, channels = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    num_faces = len(faces)

    # Detect objects (e.g., mobile phone, earphone, etc.)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Loop through YOLO detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)

    # Define objects to detect (e.g., mobile phone, earphone)
    target_objects = ['cell phone', 'remote', 'mouse', 'keyboard', 'earphone']
    mobile_phone_detected = False

    # Draw YOLO detections for target objects
    if len(indexes) > 0:
        for i in indexes.flatten():
            label = str(classes[class_ids[i]])
            if label in target_objects:
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if label == 'cell phone':
                    mobile_phone_detected = True
                    print(f"Mobile phone detected with confidence: {confidences[i]}")

    # Loop through detected faces
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = get_eye_landmarks(landmarks, left_eye_indices)
        right_eye = get_eye_landmarks(landmarks, right_eye_indices)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw the eyes
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # Check for eye contact
        if ear > eye_contact_threshold:
            cv2.putText(frame, "Eye Contact Detected", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            eye_contact_count += 1
        else:
            cv2.putText(frame, "No Eye Contact", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the number of faces
    cv2.putText(frame, f'Number of Faces: {num_faces}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Capture photo if more than one face or a mobile phone is detected
    if num_faces > 1 or mobile_phone_detected:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if num_faces > 1:
            cv2.imwrite(f"multiple_faces_{timestamp}.jpg", frame)
            print(f"Photo saved: multiple_faces_{timestamp}.jpg")
        if mobile_phone_detected:
            cv2.imwrite(f"mobile_detected_{timestamp}.jpg", frame)
            print(f"Photo saved: mobile_detected_{timestamp}.jpg")

    total_frames += 1
    if total_frames > 0:
        eye_contact_percentage = (eye_contact_count / total_frames) * 100
        print(f'Average Eye Contact Percentage: {eye_contact_percentage:.2f}%')

    # Show the video feed with detections
    cv2.imshow('Face, Eye Contact & Gadget Detector', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
