
# Age Predictor Application

This is a Python application that predicts the age of a person from an image or video using deep learning models. The application has a graphical user interface (GUI) built using the Tkinter library.

## Requirements

To run this application, you need to have the following dependencies installed:

- Python 3.x
- OpenCV
- NumPy
- PIL (Python Imaging Library)
- imutils
- tkinter

You can install the dependencies using pip with the following command:

```
pip install -r requirements.txt
```

## How to Run the Application

1. Clone the repository or download the source code files.
2. Open a terminal or command prompt and navigate to the directory where the files are located.
3. Run the following command to start the application:

```
python AgePredictor.py
```

4. The application GUI will open, allowing you to interact with the different features.

## Features

### Image Prediction

- Click on the "Image" button in the navigation menu to go to the image prediction frame.
- Click the "Open" button to select an image file.
- Once an image is selected, click the "Predict" button to predict the age of the person in the image.
- The predicted age range will be displayed on the screen along with the accuracy percentage.
- You can also click the "Save" button to save the predicted image with the age information.

### Video Prediction

- Click on the "Video" button in the navigation menu to go to the video prediction frame.
- Click the "Load video and Predict" button to select a video file.
- The video will start playing, and the application will continuously predict the age of people in the video frames.
- You can click the "Stop" button to pause the video playback.
- Click the "Save" button to save the video with the predicted age information.
- Use the "Take Snapshot" button to capture a snapshot from the video.

### Real-Time Prediction

- Click on the "Real time" button in the navigation menu to go to the real-time prediction frame.
- The application will open your webcam and start capturing live video.
- The age of people in the video frames will be continuously predicted in real-time.
- Click the "Stop" button to stop the webcam video capture.
- Use the "Take Snapshot" button to capture a snapshot from the webcam feed.

## Appearance Mode

- You can change the appearance mode of the application between "Light", "Dark", and "System" modes.
- The appearance mode menu is located at the bottom of the navigation menu.
- Selecting "Light" mode will set a light color theme for the application.
- Selecting "Dark" mode will set a dark color theme for the application.
- Selecting "System" mode will set the color theme based on your system's preferences.

## Note

- Make sure to have the necessary deep learning model files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` for face detection, and `age_deploy.prototxt` and `age_net.caffemodel` for age detection) in the respective directories (`face_detector` and `age_detector`) before running the application.
- The application may require a good processing power and GPU for real-time video processing, especially with high-resolution videos.

## Credits

This application uses the following libraries:

- OpenCV: https://opencv.org/
- NumPy: https://numpy.org/
- PIL (Python Imaging Library): https://pillow.readthedocs.io/
- imutils: https://github.com/jrosebr1/imutils
- tkinter: https://docs.python.org/3/library/tk

## Limitations

The Age Predictor App has a few limitations that should be noted:

1. Accuracy: The age prediction provided by the app is based on the information available in the input text. While the model tries to make accurate predictions, it may not always be correct, especially if the input text is ambiguous or insufficient.

2. Data Dependency: The age predictor relies on the data it has been trained on, which may have certain biases or limitations. The accuracy of the predictions can be influenced by the quality and diversity of the training data.

3. Generalization: The age predictor may not perform well when presented with input text that is significantly different from the training data. It may struggle with understanding specific cultural references, domain-specific jargon, or slang that is not well-represented in its training corpus.

4. Interpretability: The model's predictions are generated based on complex algorithms and patterns learned from the training data. It may not be possible to provide a detailed explanation or justification for how the model arrived at a particular prediction.

5. Privacy and Security: When using the Age Predictor App, it's important to be cautious with the information provided in the input text. While we make efforts to protect user data, there is always a risk of potential privacy breaches or unauthorized access.

6. Legal and Ethical Considerations: The app should be used responsibly, following applicable laws and ethical guidelines. It is essential to be mindful of the potential impact of the predictions and to avoid using the app for any malicious or harmful purposes.

Please keep these limitations in mind when using the Age Predictor App, and understand that the predictions should be taken as estimates rather than definitive statements.
