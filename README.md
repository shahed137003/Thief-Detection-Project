# 🕵️‍♂️ Thief Detection Project 🚨

## 📖 Overview

This project aims to build a machine learning-based system that detects the presence of a thief in a monitored environment, such as a home, office, or public area. The system uses surveillance camera footage or sensor data and applies machine learning techniques to classify unusual or suspicious behavior, helping security personnel take appropriate action.

## ✨ Features

- **Real-time Detection** ⏱️: Detects suspicious movements or behaviors in real time.
- **Pre-trained Model** 🤖: Uses a pre-trained model (such as a Convolutional Neural Network) for efficient processing of surveillance footage.
- **Alert System** 📲: Sends notifications or alerts when a possible intrusion is detected.
- **Scalability** 🌐: The system can be scaled to monitor multiple locations or areas simultaneously.
- **Low False-Positive Rate** 📉: Uses advanced techniques to minimize false alarms.

## 🗂️ Project Structure

```bash
thief-detection-project/
│
├── Shop DataSet/
├── models/
│   └── model.pth  # Pre-trained machine learning model
│
├── notebooks/
│   └── pretrained video classification.ipynb    # Jupyter notebook for model training
│   └── thief detection model.ipynb  # Jupyter notebook for model evaluation
│
├── VideoClassificationDeployment/
|   └── media  # contains the saved videos used for testing the model
|   └──templates # contains the ront end with style (home.html)
│   └── VideoClassificationDeployment #contains the main files.py
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── thief_detection.py     # Main program to run the detection
```
## 🗃️ Dataset

The system uses an image or video dataset consisting of two categories:

1. **Normal Behavior** 👥: This includes footage where no suspicious or unusual activity is happening.
2. **Suspicious Behavior** 🕶️: This includes simulated thief behavior like breaking into homes, stealing, or sneaking into restricted areas.

The dataset must be labeled and split into training and test sets for the machine learning model to learn effectively.

---

## 🔄 Data Preprocessing

To prepare the dataset for training, the following preprocessing steps are applied:

- **Resizing**: All images or video frames are resized to a fixed size (e.g., 128x128 pixels) to ensure consistency across the input data.
- **Normalization**: Pixel values are normalized to a range of 0-1 to enhance model performance.
- **Splitting**: The dataset is divided into training, validation, and test sets.

Preprocessing is done using a Python script:

```bash
python scripts/preprocess_data.py
```
## 🧠 Model Architecture

The thief detection system uses a **Convolutional Neural Network (CNN)** for image classification. CNNs are particularly effective for image recognition tasks. The architecture consists of the following key components:

- **Conv2D layers**: Used for feature extraction from the input images, capturing spatial hierarchies.
- **MaxPooling layers**: Reduces the dimensionality of the features while retaining important information.
- **Dense layers**: Fully connected layers for classification.
- **Softmax activation**: Used for output classification, predicting whether an input image is normal or suspicious.

You can view and modify the architecture in the `training.ipynb` notebook.

---

## 🚀 Usage

To use the thief detection system, you can run the detection model on either live or recorded video footage.

### Running the detection on a video:

```bash
python thief_detection.py --input_path path_to_video_file --model_path models/thief_detection_model.h5
```

