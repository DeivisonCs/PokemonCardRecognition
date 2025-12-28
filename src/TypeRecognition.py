import threading
import pandas as pd
import time

class TypeRecognition:
    def __init__(self, dataset_path):
        self.lock_control = threading.Lock()
        self.dataset_path = dataset_path

    def _searching_text(self):
        try:
            print("--- Thread: Iniciando trabalho real ---")
            time.sleep(5)
        finally:
            # O 'finally' garante que, mesmo se der erro, o lock será solto
            print("--- Thread: Trabalho feito, liberando lock ---")
            self.lock_control.release()

    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)

    def detect_type(self):
        if self.lock_control.acquire(blocking=False):
            print("Loop: Lock conseguido! Disparando thread...")
            t = threading.Thread(target=self._searching_text)
            t.start()
        else:
            print("Loop: Thread ainda ocupada...")
    
        time.sleep(1)