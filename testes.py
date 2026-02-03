import threading
import time
import cv2
import pandas as pd

# O objeto de controle
# LOCK_CONTROL = threading.Lock()

# def minha_tarefa():
#     try:
#         print("--- Thread: Iniciando trabalho real ---")
#         time.sleep(5)  # Simulando trabalho
#     finally:
#         # O 'finally' garante que, mesmo se der erro, o lock será solto
#         print("--- Thread: Trabalho feito, liberando lock ---")
#         LOCK_CONTROL.release()

# while True:
#     # Tenta pegar o lock. Se conseguir, dispara a thread.
#     if LOCK_CONTROL.acquire(blocking=False):
#         print("Loop: Lock conseguido! Disparando thread...")
#         t = threading.Thread(target=minha_tarefa)
#         t.start()
#     else:
#         print("Loop: Thread ainda ocupada...")

#     time.sleep(1)

# import pandas as pd

# Substitua 'pokemon.csv' pelo caminho correto do seu arquivo CSV
df = pd.read_csv('./dataset/pokemon.csv')

# Exibir as primeiras linhas do dataset para verificar os dados
result = df[df['name'].str.lower() == 'pikachu']
if not result.empty:
    nome = result['name'].values[0]
    tipo1 = result['type1'].values[0]
    tipo2 = result['type2'].values[0] if pd.notnull(result['type2'].values[0]) else None
    
    print(f"Name: {nome}\nType1: {tipo1}\nType2: {tipo2}")
else:
    print("empty")

# image = cv2.imread("./dataset/cards/Cards01.jpg")
# xmin, ymin, xmax, ymax = 100, 100, 300, 300
# frame_recortado = image[ymin:ymax, xmin:xmax]

# cv2.imshow("teset", image)
# cv2.waitKey(0)
# cv2.imshow("teset", frame_recortado)
# cv2.waitKey(0)
# cv2.destroyAllWindows()