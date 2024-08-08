import cv2
from deepface import DeepFace
import os
font = cv2.FONT_HERSHEY_SIMPLEX


input_dir = ''
output_dir = ''

# Загрузка классификатора для детекции лиц
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Покадровая детекция лица
for i in image_files:
    image_path = os.path.join(input_dir, i)
    image = cv2.imread(image_path)

    # При помощи модели DeepFace анализируем лицо и получаем список параметров
    # из которых выберем расовую принадлежность
    try:
        res = DeepFace.analyze(image, actions=['race', 'age', 'gender'], enforce_detection=False)
    except:
        print("no face")
        continue

    # Рисуем прямоугольник вокруг лица с помощью каскадов Хаара
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4, None, (200, 200))
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        print(res)

        # Выводим на картинку результаты классификации
        asian = int(res[0]['race']['asian'])
        black = int(res[0]['race']['black'])
        white = int(res[0]['race']['white'])
        gender = res[0]['dominant_gender']
        age = int(res[0]['age'])

        cv2.putText(image, str(f'Asian: {asian} %'), (x-110, y+10), font, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(image, str(f'Black: {black} %'), (x-110, y+35), font, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(image, str(f'White: {white} %'), (x-110, y+60), font, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(image, str(f'Gender: {gender} '), (x - 110, y + 85), font, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(image, str(f'Age: {age}'), (x-110, y + 85), font, 1, (255, 255, 255), 2, cv2.LINE_4)

    # Полученный результат
    cv2.imwrite(os.path.join(output_dir, i), image)