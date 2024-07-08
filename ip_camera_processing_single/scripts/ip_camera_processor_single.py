#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import torch
from torchvision import models, transforms
from PIL import Image as PILImage
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import csv
import os
from pynvml import *

# Variables for easy configuration
NUM_CLASSES = 205
POWER_MONITORING_MODE = 'gpu'  # 'gpu' to monitor GPU power usage
SHUTDOWN_TIME = 300  # Time in seconds to terminate the program (e.g., 5 minutes)
MODEL_PATH_TEMPLATE = '/home/ganeshvar/catkin_ws/src/models/{}_model_epoch_19.pth'
LOG_FILE_PATH = '/home/ganeshvar/catkin_ws/src/ip_camera_processing_single/logs/predictions_log.csv'

# Functions to handle the model and predictions
def get_class_name(class_number):
    data = pd.read_csv('/home/ganeshvar/catkin_ws/src/ip_camera_processing_single/files.csv')
    filtered_data = data[data['class'] == class_number]
    if not filtered_data.empty:
        return filtered_data.iloc[0]['file_name'].split('/')[1]
    return "Unknown"

def create_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError("Invalid model name")
    return model

def load_model_for_inference(model_path, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name)
    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(cv_image):
    img = PILImage.fromarray(cv_image)
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

class IPCameraProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.model_name = rospy.get_param('~model_name', 'resnet18')
        self.model_path = rospy.get_param('~model_path', MODEL_PATH_TEMPLATE.format(self.model_name))
        video_url = rospy.get_param('~video_url')  # Use the IP address of your DroidCam

        self.model = load_model_for_inference(self.model_path, self.model_name)
        self.result_pub = rospy.Publisher('/image_processing/result', String, queue_size=10)
        
        # CSV logging setup
        self.csv_file = LOG_FILE_PATH
        self.setup_csv()
        
        self.capture = cv2.VideoCapture(video_url)
        
        # Power measurement setup
        self.gpu_power_usage = 0.0
        self.gpu_power_supported = False
        
        if POWER_MONITORING_MODE == 'gpu':
            try:
                nvmlInit()
                self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
                self.gpu_power_supported = True
            except NVMLError as err:
                self.gpu_power_supported = False
                self.gpu_handles = []
        
        # Termination after configured time
        rospy.Timer(rospy.Duration(SHUTDOWN_TIME), self.shutdown)

    def setup_csv(self):
        # Create directory if not exists
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        # Check if the file is empty and write headers
        if not os.path.isfile(self.csv_file) or os.path.getsize(self.csv_file) == 0:
            with open(self.csv_file, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Timestamp', 'Model Name', 'Prediction', 'Processing Time (s)', 'GPU Power (W)'])

    def log_to_csv(self, timestamp, model_name, prediction, processing_time, gpu_power):
        with open(self.csv_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([timestamp, model_name, prediction, processing_time, gpu_power])

    def process_frames(self):
        while not rospy.is_shutdown():
            ret, frame = self.capture.read()
            if not ret:
                continue

            image = preprocess_image(frame)
            
            start_time = time.time()
            
            # Initialize power measurement
            gpu_power_start = 0
            if self.gpu_power_supported:
                try:
                    gpu_power_start = sum([nvmlDeviceGetPowerUsage(handle) for handle in self.gpu_handles])
                except NVMLError as err:
                    pass
            
            prediction = predict_image(self.model, image)
            
            # Finalize power measurement
            gpu_power_end = 0
            if self.gpu_power_supported:
                try:
                    gpu_power_end = sum([nvmlDeviceGetPowerUsage(handle) for handle in self.gpu_handles])
                except NVMLError as err:
                    pass

            end_time = time.time()
            total_time = end_time - start_time
            gpu_power_usage = (gpu_power_end - gpu_power_start) / 1000  # Convert from mW to W
            
            # Get class name
            class_name = get_class_name(prediction)
            
            # Result
            result = f'Prediction: {class_name}, Time: {total_time:.2f}s'
            self.result_pub.publish(result)
            
            # Log to CSV
            timestamp = rospy.get_time()
            self.log_to_csv(timestamp, self.model_name, class_name, total_time, gpu_power_usage)
            
            # ROS log in the specified format
            rospy.loginfo(f"{timestamp},{self.model_name},{class_name},{total_time},{gpu_power_usage}")

    def shutdown(self, event):
        rospy.signal_shutdown("Node shutdown")

def predict_image(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

def main():
    rospy.init_node('ip_camera_processor', anonymous=True)
    ip = IPCameraProcessor()
    ip.process_frames()
    rospy.spin()

if __name__ == '__main__':
    main()
