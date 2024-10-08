import cv2
import numpy as np
from keras.models import load_model # type: ignore
import time
from collections import Counter

# Correct path usage
model_path = r'C:\Users\SOAIB ASLAM\OneDrive\Desktop\PROJECT\Facial Expression Recognition System\emotion detection\model.h5'
model = load_model(model_path)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the haarcascade for face detection
face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

# Initialize variables for FPS calculation and emotion statistics
fps_counter = 0
fps_start_time = time.time()
emotion_count = Counter()

# Initialize pause/resume functionality
paused = False
pause_start_time = None
resume_start_time = None

# Function to detect and predict emotions
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)  # Adjusted parameters
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
        
        # Prediction
        try:
            emotion_prediction = model.predict(reshaped_face)
            print("Emotion probabilities:", emotion_prediction)  # Debugging output
            emotion_index = np.argmax(emotion_prediction)
            emotion_label = emotion_labels[emotion_index]
            emotion_count[emotion_label] += 1
        except Exception as e:
            print(f"Error during prediction: {e}")
            emotion_label = 'Unknown'
        
        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image

# Capture video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Call the function to detect emotions
        result = detect_emotion(frame)
        
        # Calculate FPS
        fps_counter += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            fps_start_time = time.time()
            fps_text = f'FPS: {fps:.2f}'
        else:
            fps_text = f'FPS: {fps:.2f}'
        
        # Display FPS
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display Emotion Statistics
        stats_text = 'Emotions: ' + ', '.join([f'{emotion}: {count}' for emotion, count in emotion_count.items()])
        cv2.putText(result, stats_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Display Pause/Resume Status
        status_text = 'Paused' if paused else 'Running'
        cv2.putText(result, status_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if paused else (0, 255, 0), 2, cv2.LINE_AA)
    
        cv2.imshow('Emotion Detection', result)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the current frame with the detected emotion
        filename = f'emotion_{int(time.time())}.png'
        cv2.imwrite(filename, result)
        print(f'Saved {filename}')
    elif key == ord('p'):
        # Pause/Resume functionality
        if paused:
            paused = False
            resume_start_time = time.time()
            if pause_start_time:
                pause_duration = resume_start_time - pause_start_time
                print(f'Paused for {pause_duration:.2f} seconds')
        else:
            paused = True
            pause_start_time = time.time()
            print('Paused')
    
# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
