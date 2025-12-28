import threading
import os
import cv2
import pandas as pd

from ultralytics import YOLO

class TypeRecognition:
    def __init__(self, model_path="my_model.pt", dataset_path="./dataset/pokemon.csv"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.border_box_color = (164,120,87)
        self.lock_control = threading.Lock()
        self.model = None
        self.labels = None
        self.min_confidence = 0.8

    def searching_text(self):
        try:
            print("--- Thread: Iniciando trabalho real ---")
        finally:
            # O 'finally' garante que, mesmo se der erro, o lock será solto
            print("--- Thread: Trabalho feito, liberando lock ---")
            self.lock_control.release()

    def load_model(self):
        if (not os.path.exists(self.model_path)):
            print('ERROR: Incorrect or invalid path.')
            return

        self.model = YOLO(self.model_path, task='detect')
        self.labels = self.model.names
        print("Model loaded")

    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)

    def get_item_coordinates(self, detection):
        xyxy_tensor = detection.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        return xyxy.astype(int)

    def get_class_name(self, detection):
        classidx = int(detection.cls.item())
        return self.labels[classidx]

    def detect_item_on_frame(self, frame, verbose=False):
        results = self.model(frame, verbose=verbose)
        detections = results[0].boxes

        for detection in detections:
            xmin, ymin, xmax, ymax = self.get_item_coordinates(detection)
            classname = self.get_class_name(detection)
            confidence = detection.conf.item()

            if confidence > self.min_confidence:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), self.border_box_color, 2)

                label = f'{classname}: {int(confidence*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10) # Making sure label is not to close to top
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), self.border_box_color, cv2.FILLED) # Draw white box to put
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)