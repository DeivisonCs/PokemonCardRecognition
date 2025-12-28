import cv2
import pandas as pd
import ObjectRecognition
import TypeRecognition

CAM_WIDTH, CAM_HEIGHT = 680, 480
CAM_FPS = 60
DATASET_PATH = "./dataset/pokemon.csv"

def draw_last_results(frame, item):
    if item is not None:
        cv2.putText(frame, "Last Pokemon detected:", (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.putText(frame, f"Name: {item["name"]}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.putText(frame, f"Type 1: {item["type1"]}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        
        if item["type2"]:
            cv2.putText(frame, f"Type 2: {item["type2"]}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

def main():
    print("App started")
    object_recognition = ObjectRecognition.ObjectRecognition()
    object_recognition.load_model()

    type_recognition = TypeRecognition.TypeRecognition(dataset_path=DATASET_PATH)
    type_recognition.load_dataset()

    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    cap.set(5, CAM_FPS)

    if not cap.isOpened():
        print("Couldn't access camera")
        exit()

    while True:
        success, frame = cap.read()
        objects_detect_result = object_recognition.detect_object_on_frame(frame)
        objects_box_detected = objects_detect_result["objects_box_detected"]

        for object_box in objects_box_detected:
            xmin, xmax = object_box["xmin"], object_box["xmax"]
            ymin, ymax = object_box["ymin"], object_box["ymax"]
            item_detected = type_recognition.detect_text_in_frame(frame[ymin:ymax, xmin:xmax])
            
            if item_detected is not None:
                name = item_detected['name'].values[0]
                type_1 = item_detected['type1'].values[0]
                type_2 = item_detected['type2'].values[0] if pd.notnull(item_detected['type2'].values[0]) else None
                type_recognition.last_item_found = {"name":name, "type1":type_1, "type2":type_2}
                
                print("Pokemon found")
                print(f"Name: {name}")
                print(f"Type 1: {type_1}")
                print(f"Type 2: {type_2}")
            
            draw_last_results(frame, type_recognition.last_item_found)

        cv2.imshow("App Cam", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("App finished")

if __name__ == "__main__":
    main()