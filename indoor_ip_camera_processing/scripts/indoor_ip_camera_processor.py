#!/usr/bin/env python3

import os
import time
import csv
import cv2
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError

# Set a seed for reproducibility in initial data splitting
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

# Get the number of available CPU cores
num_cores = multiprocessing.cpu_count()

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for the validation and test datasets without data augmentation
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, indoor_classes=None):
        self.annotations = pd.read_csv(csv_file)
        self.annotations['class_name'] = self.annotations['file_name'].apply(lambda x: x.split('/')[1])
        if indoor_classes:
            self.annotations = self.annotations[self.annotations['class_name'].isin(indoor_classes)]
        self.root_dir = root_dir
        self.transform = transform

        # Verify and set the number of classes
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(self.annotations['class_name'].unique()))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Update labels to class indices
        self.annotations['class_idx'] = self.annotations['class_name'].map(self.class_to_idx)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        label = int(self.annotations.iloc[index, self.annotations.columns.get_loc('class_idx')])

        if self.transform:
            image = self.transform(image)

        return (image, label)

# Define indoor classes
indoor_classes = [
    'motel', 'hotel', 'kitchenette', 'coffee_shop', 'kitchen', 'television_studio', 
    'inn', 'classroom', 'parlor', 'dinette', 'laundromat', 'basilica', 'bookstore', 
    'restaurant', 'apartment_building', 'art_gallery', 'living_room', 'ballroom', 
    'engine_room', 'beauty_salon', 'music_studio', 'closet', 'attic', 'martial_arts_gym', 
    'basement', 'bowling_alley', 'hotel_room', 'shoe_shop', 'mansion', 'gift_shop', 
    'waiting_room', 'fire_station', 'reception', 'dining_room', 'clothing_store', 
    'abbey', 'gas_station', 'ice_skating_rink', 'nursery', 'cafeteria', 'art_studio', 
    'bakery', 'jail_cell', 'cockpit', 'restaurant_kitchen', 'conference_center', 
    'food_court', 'ice_cream_parlor', 'boxing_ring', 'train_station', 'bus_interior', 
    'auditorium', 'schoolhouse', 'bar', 'pantry', 'hospital', 'corridor', 
    'office_building', 'supermarket', 'galley', 'game_room', 'banquet_hall', 'shower', 
    'locker_room', 'airport_terminal', 'bedroom', 'subway_station', 'aquarium', 
    'museum', 'home_office', 'hospital_room', 'kindergarden_classroom', 'butchers_shop', 
    'candy_store', 'office', 'conference_room', 'assembly_line', 'dorm_room'
]

# Load the dataset
csv_file = '/home/ganeshvar/catkin_ws/src/indoor_ip_camera_processing/files.csv'
root_dir = '/home/ganeshvar/catkin_ws/src/indoor_ip_camera_processing/'
dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=val_test_transform, indoor_classes=indoor_classes)

# Function to create models
def create_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Invalid model name")
    return model


# Function to load the models
def load_model(model_path, model_name, num_classes):
    model = create_model(model_name, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if the model was saved with DataParallel
    if next(iter(state_dict)).startswith('module.'):
        # Remove the 'module.' prefix
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore the fc layer mismatch
    
    model = model.to(device)
    
    # Wrap the model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.eval()
    return model

# Function to preprocess image
def preprocess_image(cv_image):
    img = Image.fromarray(cv_image)
    if img.size != (256, 256):
        img = img.resize((256, 256))
    img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(img)
    return img_t.unsqueeze(0)

# IPCameraProcessor class
class IPCameraProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.model_name = rospy.get_param('~model_name', 'resnet50')
        self.model_path = rospy.get_param('~model_path', f'/home/ganeshvar/catkin_ws/src/models/indoor_{self.model_name}_epoch_20.pth')
        self.model = load_model(self.model_path, self.model_name, num_classes=len(indoor_classes))
        self.result_pub = rospy.Publisher('/image_processing/result', String, queue_size=10)
        
        # CSV logging setup
        self.csv_file = rospy.get_param('~csv_file', '/home/ganeshvar/catkin_ws/src/indoor_ip_camera_processing/logs/predictions_log.csv')
        self.setup_csv()
        
        video_url = rospy.get_param('~video_url')  # Use the IP address of your DroidCam
        self.capture = cv2.VideoCapture(video_url)
        
        # GPU power monitoring setup
        nvmlInit()
        self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

    def setup_csv(self):
        # Create directory if not exists
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        # Check if the file is empty and write headers
        if not os.path.isfile(self.csv_file) or os.path.getsize(self.csv_file) == 0:
            with open(self.csv_file, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Timestamp', 'Model Name', 'Prediction', 'Processing Time (s)', 'GPU Power (W)'])

    def log_to_csv(self, timestamp, model_name, prediction, processing_time, gpu_power_usage):
        with open(self.csv_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([timestamp, model_name, prediction, processing_time, gpu_power_usage])

    def process_frames(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            if time.time() - start_time > 300:  # 5 minutes = 300 seconds
                rospy.signal_shutdown("Node shutdown after 5 minutes")
                break

            ret, frame = self.capture.read()
            if not ret:
                rospy.logerr("Failed to capture image")
                continue

            image = preprocess_image(frame)
            frame_start_time = time.time()
            gpu_power_start = sum([nvmlDeviceGetPowerUsage(handle) for handle in self.gpu_handles]) if self.gpu_handles else 0
            
            prediction = predict_image(self.model, image)
            
            gpu_power_end = sum([nvmlDeviceGetPowerUsage(handle) for handle in self.gpu_handles]) if self.gpu_handles else 0
            gpu_power_usage = (gpu_power_end - gpu_power_start) / 1000  # Convert from mW to W
            
            frame_end_time = time.time()
            total_time = frame_end_time - frame_start_time
            class_name = get_class_name(prediction)
            
            result = f'Prediction: {class_name}, Time: {total_time:.2f}s, GPU Power: {gpu_power_usage:.3f}W'
            self.result_pub.publish(result)
            rospy.loginfo(result)
            
            # Log to CSV
            self.log_to_csv(rospy.get_time(), self.model_name, class_name, total_time, gpu_power_usage)

def predict_image(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

def get_class_name(class_number):
    data = pd.read_csv('/home/ganeshvar/catkin_ws/src/indoor_ip_camera_processing/files.csv')
    filtered_data = data[data['class'] == class_number]
    if not filtered_data.empty:
        return filtered_data.iloc[0]['file_name'].split('/')[1]
    return "Unknown"

def main():
    rospy.init_node('indoor_ip_camera_processor', anonymous=True)
    ip = IPCameraProcessor()
    ip.process_frames()
    rospy.spin()

if __name__ == '__main__':
    main()

