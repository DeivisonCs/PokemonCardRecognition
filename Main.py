import os
import sys

import cv2
from ultralytics import YOLO

CAM_WIDTH, CAM_HEIGHT = 680, 480
CAM_FPS = 100
MODEL_PATH = "my_model.pt"
BORDER_BOX_COLOR = (164,120,87)

def load_model(model_path:str):
    if (not os.path.exists(model_path)):
        print('ERROR: Incorrect or invalid path.')
        sys.exit(0)

    model = YOLO(MODEL_PATH, task='detect')
    labels = model.names
    print("Model loaded")

    return model, labels

def main():
    print("App started")
    model, labels = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    cap.set(5, CAM_FPS)

    if not cap.isOpened():
        print("Couldn't access camera")
        exit()

    while True:
        success, frame = cap.read()

        results = model(frame, verbose=False)
        detections = results[0].boxes

        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # Bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > 0.5:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), BORDER_BOX_COLOR, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Font size
                label_ymin = max(ymin, labelSize[1] + 10) # Making sure label is not to close to top
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), BORDER_BOX_COLOR, cv2.FILLED) # Draw white box to put
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("App finished")


if __name__ == "__main__":
    main()