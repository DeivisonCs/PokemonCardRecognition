import cv2
import TypeRecognition

CAM_WIDTH, CAM_HEIGHT = 680, 480
CAM_FPS = 100

def main():
    print("App started")
    type_recognition = TypeRecognition.TypeRecognition()
    type_recognition.load_dataset()
    type_recognition.load_model()

    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    cap.set(5, CAM_FPS)

    if not cap.isOpened():
        print("Couldn't access camera")
        exit()

    while True:
        success, frame = cap.read()
        type_recognition.detect_item_on_frame(frame)

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("App finished")

if __name__ == "__main__":
    main()