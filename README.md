Facial_Expression_Recognition_System# Emotion Detection System

This Emotion Detection System uses a webcam to capture real-time facial expressions and detect emotions using a pre-trained deep learning model. It displays the predicted emotions on-screen and provides additional features such as emotion statistics, FPS display, and pause/resume functionality.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup and Requirements](#setup-and-requirements)
- [Usage](#usage)
- [Functionality](#functionality)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project detects seven basic emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. It uses OpenCV for face detection and a Keras-trained model to predict emotions from facial expressions. The real-time video processing provides a fluid experience and includes FPS display and emotion tracking.

## Features

- **Real-Time Emotion Detection**: Captures video and predicts emotions in real time.
- **Emotion Statistics**: Tracks occurrences of each detected emotion.
- **FPS Display**: Displays the frames per second (FPS) rate.
- **Pause/Resume Functionality**: Allows pausing/resuming emotion detection using keyboard shortcuts.
- **Screenshot Capture**: Saves current frame with detected emotions.

## Setup and Requirements

1. **Install Requirements**:
   - Install necessary Python libraries:
     ```bash
     pip install opencv-python-headless numpy keras
     ```
2. **Download Model**:
   - Place your pre-trained model (`model.h5`) at the specified `model_path` in the code.

3. **Directory Structure**:
   - Make sure your `haarcascade_frontalface_default.xml` file is accessible by OpenCV from the default path:
     ```plaintext
     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
     ```

## Usage

1. **Run the Program**:
   - Execute the script:
     ```bash
     python emotion_detection.py
     ```
2. **Keyboard Controls**:
   - **'q'**: Quit the application.
   - **'s'**: Save a screenshot of the current frame.
   - **'p'**: Toggle Pause/Resume.

## Functionality

- **Emotion Prediction**: Detects faces in each frame, preprocesses them, and uses a deep learning model to predict emotions.
- **Display**:
  - Draws a rectangle around the detected face and labels it with the predicted emotion.
  - Displays FPS, emotion statistics, and status (Paused/Running).
  
## Error Handling

- **Face Detection**: If no face is detected, the application continues without any output until a face is visible.
- **Model Prediction**: If an error occurs during emotion prediction, an error message will be displayed on the console, and the detected emotion will default to 'Unknown'.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add a new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License.

[Facial Expression Recognition System Code](https://github.com/SoaibAslam/Facial_Expression_Recognition_System/blob/main/test.py)
