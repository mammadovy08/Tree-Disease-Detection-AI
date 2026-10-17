[kamera.py](https://github.com/user-attachments/files/33606042/kamera.py)
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


def sriftı_tap(olcu=26):
    for yol in ["C:/Windows/Fonts/arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(yol, olcu)
        except OSError:
            continue
    return ImageFont.load_default()


SRIFT = sriftı_tap(26)
SRIFT2 = sriftı_tap(18)


def yaz(frame, metn, reng, metn2=""):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 76], fill=(0, 0, 0))
    d.text((10, 6), metn, font=SRIFT, fill=reng)
    d.text((10, 46), metn2, font=SRIFT2, fill=(200, 200, 200))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def saglamdir(ad):
    a = ad.lower()
    return "sağlam" in a or "saglam" in a or "health" in a


#Model və sinif adları
model = tf.keras.models.load_model("keras_model.h5", compile=False)

classes = []
with open("labels.txt", encoding="utf-8-sig") as f:
    for sətir in f:
        sətir = sətir.strip()
        if not sətir:
            continue
        hisse = sətir.split(" ", 1)          # "0 Unlu şeh" -> ["0", "Unlu şeh"]
        classes.append(hisse[1] if len(hisse) == 2 else hisse[0])

print("Siniflər:", classes)

#Kamera
cap = cv2.VideoCapture(0)       
if not cap.isOpened():
    raise SystemExit("Kamera açılmadı. Başqa proqram kameranı işlədir ola bilər.")

print("Çıxmaq üçün Q, şəkli saxlamaq üçün S düyməsini basın")

kadr_sayi = 0
metn, reng = "Yarpağı çərçivənin içinə qoyun", (255, 255, 255)
metn2 = ""

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    k = min(h, w)
    y0, x0 = (h - k) // 2, (w - k) // 2

    kadr_sayi += 1
    if kadr_sayi % 5 == 0:      # hər 5-ci kadrı yoxla
        kvadrat = frame[y0:y0 + k, x0:x0 + k]
        rgb = cv2.cvtColor(cv2.resize(kvadrat, (224, 224)), cv2.COLOR_BGR2RGB)
        x = np.expand_dims(rgb.astype("float32") / 127.5 - 1, axis=0)

        pred = model.predict(x, verbose=0)[0]
        i = int(np.argmax(pred))
        faiz = float(pred[i]) * 100

        top = np.argsort(pred)[::-1][1:3]
        metn2 = "Digər ehtimallar: " + " | ".join(
            f"{classes[j]} {pred[j]*100:.0f}%" for j in top)

        if faiz < 80:
            metn, reng = f"Əmin deyiləm ({faiz:.0f}%) - yarpağa yaxınlaşın", (255, 200, 0)
        else:
            metn = f"{classes[i]} ({faiz:.0f}%)"
            reng = (0, 255, 0) if saglamdir(classes[i]) else (255, 80, 80)

    ekran = frame.copy()
    cv2.rectangle(ekran, (x0, y0), (x0 + k, y0 + k), (255, 255, 255), 2)
    cv2.imshow("Yarpaq tanima", yaz(ekran, metn, reng, metn2))

    duyme = cv2.waitKey(1) & 0xFF
    if duyme == ord("q"):
        break
    if duyme == ord("s"):
        cv2.imwrite("cekilen_sekil.jpg", frame)
        print("Şəkil saxlandı: cekilen_sekil.jpg")

cap.release()
cv2.destroyAllWindows()
