import threading
import pandas as pd
import cv2
import easyocr
import queue
import re

class TypeRecognition:
    def __init__(self, dataset_path, gpu=False):
        self.lock_control = threading.Lock()
        self.dataset_path = dataset_path
        self.dataset = None
        self.reader = easyocr.Reader(["pt"], gpu=gpu)

    def _clean_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)

    def _search_item_in_dateaset(self, text):
        cleaned_text = self._clean_text(text)
        result = self.dataset[self.dataset['name'].str.lower() == cleaned_text.lower()]

        if not result.empty:
            return result
        else:
            return None

    def _searching_text(self, **kwargs):
        try:
            frame = kwargs.get("frame", None)
            result_queue = kwargs.get("result_queue", None)
            
            if frame is None or frame.size == 0 or not result_queue:
                return

            texts_detected = self.reader.readtext(frame)
            for text in texts_detected:
                item = self._search_item_in_dateaset(text[1])

                if item is not None:
                    result_queue.put(item)
        finally:
            self.lock_control.release()

    def detect_text_in_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.lock_control.acquire(blocking=False):
            print("Procurando...")
            result_queue = queue.Queue()

            thread = threading.Thread(target=self._searching_text, kwargs={"frame": gray_frame, "result_queue": result_queue})
            thread.start()
            thread.join()

            if not result_queue.empty():
                result = result_queue.get()
                return result
        
        return None