import cv2
import ObjectRecognition
import TypeRecognition

CAM_WIDTH, CAM_HEIGHT = 680, 480
CAM_FPS = 60
DATASET_PATH = "./dataset/pokemon.csv"

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
            type_recognition.detect_text_in_frame(frame[ymin:ymax, xmin:xmax])

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("App finished")

if __name__ == "__main__":
    main()