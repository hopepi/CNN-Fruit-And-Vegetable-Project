import tensorflow as tf
from keras.src.saving import load_model
from keras.src.utils.image_utils import load_img, img_to_array

import numpy as np
from pathlib import Path


# Modeli yükleme
model = load_model('my_model.h5')


base_dir = Path("C:/Users/umutk/OneDrive/Masaüstü/newCnnProject/myTest")
img_path1 = base_dir / "test1.jpg"
img_path2 = base_dir / "test2.jpg"
img_path3 = base_dir / "test3.jpg"
img_path4 = base_dir / "test4.jpg"
img_path5 = base_dir / "test5.jpg"
img_path6 = base_dir / "test6.jpg"
img_path7 = base_dir / "test7.jpg"
img_path8 = base_dir / "test8.jpg"

# Görüntüleri yükleme ve ön işleme
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)  # Diziye çevir
    img_array = np.expand_dims(img_array, axis=0)  # Boyutunu genişlet
    img_array /= 255.0  # Normalleştir
    return img_array

img_array1 = preprocess_image(img_path1)
img_array2 = preprocess_image(img_path2)
img_array3 = preprocess_image(img_path3)
img_array4 = preprocess_image(img_path4)
img_array5 = preprocess_image(img_path5)
img_array6 = preprocess_image(img_path6)
img_array7 = preprocess_image(img_path7)
img_array8 = preprocess_image(img_path8)

# Tahmin yapma
predictions1 = model.predict(img_array1)
predicted_class1 = np.argmax(predictions1, axis=1)  # Tahmin edilen sınıfı al
print(f'Test1 = Tahmin edilen sınıf: {predicted_class1}')

predictions2 = model.predict(img_array2)
predicted_class2 = np.argmax(predictions2, axis=1)  # Tahmin edilen sınıfı al
print(f'Test2 = Tahmin edilen sınıf: {predicted_class2}')

predictions3 = model.predict(img_array3)
predicted_class3 = np.argmax(predictions3, axis=1)  # Tahmin edilen sınıfı al
print(f'Test3 = Tahmin edilen sınıf: {predicted_class3}')

predictions4 = model.predict(img_array4)
predicted_class4 = np.argmax(predictions4, axis=1)  # Tahmin edilen sınıfı al
print(f'Test4 = Tahmin edilen sınıf: {predicted_class4}')

predictions5 = model.predict(img_array5)
predicted_class5 = np.argmax(predictions5, axis=1)  # Tahmin edilen sınıfı al
print(f'Test5 = Tahmin edilen sınıf: {predicted_class5}')

predictions6 = model.predict(img_array6)
predicted_class6 = np.argmax(predictions6, axis=1)  # Tahmin edilen sınıfı al
print(f'Test6 = Tahmin edilen sınıf: {predicted_class6}')

predictions7 = model.predict(img_array7)
predicted_class7 = np.argmax(predictions7, axis=1)  # Tahmin edilen sınıfı al
print(f'Test7 = Tahmin edilen sınıf: {predicted_class7}')

predictions8 = model.predict(img_array8)
predicted_class8 = np.argmax(predictions8, axis=1)  # Tahmin edilen sınıfı al
print(f'Test8 = Tahmin edilen sınıf: {predicted_class8}')

