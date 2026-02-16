# Vehicle Tracking and Counting using YOLOv8n

## Overview
This project implements a real-time **vehicle tracking and counting system** using the **YOLOv8n** object detection model.  
It detects cars from video footage, tracks them across frames, and counts vehicles as they cross a defined virtual line.

## Features
- Real-time vehicle detection using YOLOv8n  
- Vehicle tracking across video frames  
- Automatic counting with line-crossing logic  
- Works on recorded traffic videos

## Tech Stack
- Python  
- Ultralytics YOLOv8  
- OpenCV  
- NumPy  

## How It Works
1. Input video is processed frame-by-frame  
2. YOLOv8n detects vehicles  
3. Tracking assigns IDs to vehicles  
4. Vehicles crossing the counting line are counted

# Screenshots

![image alt](https://github.com/sanjaiycs/Car-Tracking-Using-YOLO/blob/d9300929c0deccddeaf9ddf33a9242eb440328ae/detection.png)

![image alt](https://github.com/sanjaiycs/Car-Tracking-Using-YOLO/blob/d9300929c0deccddeaf9ddf33a9242eb440328ae/d2.png)

## Installation
```bash
pip install ultralytics opencv-python numpy
```

## Usage
```bash
python main.py
```

Place your input video inside the project folder and update the video path in the script.

## Output
- Detected vehicles with bounding boxes  
- Vehicle ID tracking  
- Live count displayed on the video feed
