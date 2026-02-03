import threading
import pandas as pd
import cv2
import easyocr
import queue
import re
import time

class TypeRecognition:
    def __init__(self, dataset_path, search_interval=5, gpu=False):
        self.lock_control = threading.Lock()
        self.dataset_path = dataset_path
        self.dataset = None
        self.reader = easyocr.Reader(["pt"], gpu=gpu)
        
        self.search_interval = search_interval
        self.last_found_search = None
        self.last_item_found = None
        self.result_queue = queue.Queue()

    def _can_search(self, time):
        if self.search_interval is None:
            return True
        
        if self.last_found_search is None:
            return True
        
        elapsed_time = time - self.last_found_search
        
        if elapsed_time < self.search_interval:
            return False

        return True

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

    def _searching_text(self, frame):
        try:
            texts_detected = self.reader.readtext(frame)
            for text in texts_detected:
                item = self._search_item_in_dateaset(text[1])
                if item is not None:
                    self.result_queue.put(item)
        finally:
            self.lock_control.release()

    def detect_text_in_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()

        if self._can_search(current_time) and self.lock_control.acquire(blocking=False):
            print("Procurando...")
            self.last_found_search = current_time

            thread = threading.Thread(
                target=self._searching_text,
                args=(gray_frame,),
                daemon=True
            )
            thread.start()

        return None

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None