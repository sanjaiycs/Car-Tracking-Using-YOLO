from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

model = YOLO('/Users/sec23ad141/Desktop/yolo-model/yolo-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cap=cv2.VideoCapture("/Users/sec23ad141/Desktop/yolo-model/Videos/cars.mp4")
mask = cv2.imread("mask.png")

totalcount=[]
limits=(400,297,673,297)
tracker=Sort(max_age=20,min_hits=2,iou_threshold=0.3)
while True:
    success,img=cap.read()
    if not success:
        break
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    imgRegion = cv2.bitwise_and(img, mask_resized)
    detections=np.empty((0,5))
    result=model(imgRegion,stream=True)
    for r in result:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            conf=(math.ceil(box.conf[0]*100))/100
            cls=int(box.cls[0])
            currentclass=classNames[cls]
            if currentclass=='car' or currentclass=='bike' or currentclass=='truck' or currentclass=='bus' or currentclass=='motorcycle' and conf>0.3:
#                cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=5)
#                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.4, thickness=1,offset=3)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))
    resultTracker=tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultTracker:
        x1,y1,x2,y2,id=result
        print(result)
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=1, thickness=2,offset=3)

        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-10< cy <limits[1]+10:
            if totalcount.count(id)==0:
                totalcount.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,255),5)


        cvzone.putTextRect(img, f'count: {len(totalcount)}', (50,50), scale=1, thickness=2,offset=3)

    cv2.imshow("Image",img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
