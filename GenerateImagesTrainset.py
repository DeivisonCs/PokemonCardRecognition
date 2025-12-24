import cv2
import random
import os
import numpy as np

BG_DIR = "dataset/backgroundsTrainset"
CARD_DIR = "dataset/cards"

OUT_CLASS_PATH = "data"
OUT_CLASS_NAME = "classes"
OUT_IMG = "data/images"
OUT_LBL = "data/labels"

NUM_IMAGES = 200
CLASS_NAME= "card"
CLASS_ID = 0

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

backgrounds = os.listdir(BG_DIR)
cards = os.listdir(CARD_DIR)

def random_brightness_contrast(img):
    alpha = random.uniform(0.7, 1.3)  # contraste
    beta = random.randint(-30, 30)    # brilho
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0, 0)
    )


for i in range(NUM_IMAGES):

    bg = cv2.imread(os.path.join(BG_DIR, random.choice(backgrounds)))
    card = cv2.imread(
        os.path.join(CARD_DIR, random.choice(cards)),
        cv2.IMREAD_UNCHANGED
    )

    if bg is None or card is None:
        continue

    h_bg, w_bg, _ = bg.shape
    h_c, w_c = card.shape[:2]

    # ========================= ESCALA =========================
    min_bg = min(w_bg, h_bg)
    base_scale = (min_bg / max(w_c, h_c)) * 0.9
    scale = base_scale * random.uniform(0.25, 0.6)

    card = cv2.resize(
        card,
        (int(w_c * scale), int(h_c * scale))
    )

    # ========================= BRILHO / CONTRASTE =========================
    if card.shape[2] == 4:
        rgb = random_brightness_contrast(card[:, :, :3])
        card = np.dstack((rgb, card[:, :, 3]))
    else:
        card = random_brightness_contrast(card)

    # ========================= ROTAÇÃO =========================
    if card.shape[2] == 3:
        card = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
        card[:, :, 3] = 255

    angle = random.uniform(-35, 35)
    card = rotate_image(card, angle)

    h_c, w_c = card.shape[:2]

    # segurança
    if w_c >= w_bg or h_c >= h_bg:
        continue

    # ========================= POSIÇÃO ALEATÓRIA =========================
    x = random.randint(0, w_bg - w_c)
    y = random.randint(0, h_bg - h_c)

    # ========================= BLEND =========================
    if card.shape[2] == 4:
        card_rgb = card[:, :, :3]
        alpha = card[:, :, 3] / 255.0

        for c in range(3):
            bg[y:y+h_c, x:x+w_c, c] = (
                alpha * card_rgb[:, :, c] +
                (1 - alpha) * bg[y:y+h_c, x:x+w_c, c]
            )
    else:
        bg[y:y+h_c, x:x+w_c] = card[:, :, :3]

    # ========================= BOUNDING BOX (YOLO) =========================
    x_center = (x + w_c / 2) / w_bg
    y_center = (y + h_c / 2) / h_bg
    bw = w_c / w_bg
    bh = h_c / h_bg

    # ========================= SALVA =========================
    img_name = f"img_{i:04d}.jpg"
    cv2.imwrite(os.path.join(OUT_IMG, img_name), bg)

    label_name = img_name.replace(".jpg", ".txt")

    with open(os.path.join(OUT_CLASS_PATH, OUT_CLASS_NAME), "w") as f:
        f.write(CLASS_NAME)
    with open(os.path.join(OUT_LBL, label_name), "w") as f:
        f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

print("Dataset com escala, rotação e brilho gerado")
