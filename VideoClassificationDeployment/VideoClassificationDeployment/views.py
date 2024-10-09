import os
import torch
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from torchvision import transforms
import cv2
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from PIL import Image  # Add this import for converting frames to PIL images


# Load your trained model
model = torch.load("C:\\Users\\ADMIN\\Downloads\\VideoClassificationDeployment\\VideoClassificationDeployment\\model.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()  # Set model to evaluation mode
label_mapping = {1: 'Shoplifter', 0: 'Non-Shoplifter'}

def predict_shoplifter(video_path, model, device):
    # Preprocess the external video
    pixel_values = preprocess_external_video(video_path)

    # Move the input to the device (GPU/CPU)
    pixel_values = pixel_values.to(device)

    # Set model to evaluation mode
    model.eval()

    # Run the model with no gradient calculation (for inference)
    with torch.no_grad():
        # Make prediction
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

    # Get the predicted class (0: Non-Shoplifter, 1: Shoplifter)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to the actual label
    prediction_label = label_mapping[predicted_class]

    return prediction_label



def extract_and_sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame sampling interval
    sample_interval = max(1, total_frames // num_frames)

    count = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames at regular intervals
        if count % sample_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        count += 1

    # If fewer frames are extracted, pad with repeated frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    cap.release()
    return frames
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import torch

# Load pre-trained video transformer
feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")

def preprocess_external_video(video_path):
    # Assuming you have a function 'extract_and_sample_frames' to extract video frames
    frames = extract_and_sample_frames(video_path, num_frames=16)  # Adjust the number of frames as needed
    inputs = feature_extractor(frames, return_tensors="pt")
    return inputs['pixel_values']

def predict_video(request):
    if request.method == 'POST' and request.FILES.get('filePath', None):
        video_file = request.FILES['filePath']
        
        # Save the uploaded video
        fs = FileSystemStorage()
        video_name = fs.save(video_file.name, video_file)
        video_path = fs.path(video_name)
        
        # Preprocess the video for model input
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Convert logits to probabilities or directly get the predicted class
        predicted_class =  predict_shoplifter(video_path, model, device)
        
        # Render the result in the template
        return render(request, 'home.html', {
            'prediction': predicted_class,  # Pass the prediction result
            'video_url': fs.url(video_name)  # Pass the uploaded video URL to the template
        })
    
    return render(request, 'home.html')




