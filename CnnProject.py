import numpy as np
import pandas as pd
from pathlib import Path
import os
from keras.src.backend.jax.random import shuffle
# TensorFlow için bir çevresel değişken ayarlıyoruz, oneDNN optimizasyonlarını devre dışı bırakıyoruz
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
# Keras bileşenlerini içe aktarıyoruz
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.initializers import HeNormal
from keras.src.layers import LeakyReLU


import warnings
# Uyarıları görmezden gelmek için
warnings.filterwarnings("ignore")


# Temel dizin ve veri dizinleri tanımlanıyor
base_dir = Path("C:/Users/umutk/OneDrive/Masaüstü/newCnnProject")
train_dir = base_dir / "train"  # Eğitim verileri dizini
test_dir = base_dir / "test"  # Test verileri dizini
validation_dir = base_dir / "validation"  # Doğrulama verileri dizini


#RESİMLERİ ÇEKME FONKSİYONU
# Veri çerçevesi oluşturma fonksiyonu
def create_dataframe(directory):
    filepaths = []  # Görüntü dosyalarının yollarını tutacak liste
    labels = []  # Etiketleri tutacak liste
    # Dizin içindeki her etiket için
    for label in os.listdir(directory):
        label_dir = directory / label  # Alt dizin yolu
        if label_dir.is_dir():  # Eğer alt dizin ise
            # Alt dizindeki her görüntü dosyası için
            for img_file in os.listdir(label_dir):
                if img_file.endswith('.jpg'):  # Sadece .jpg dosyalarını al
                    filepaths.append(str(label_dir / img_file))  # Dosya yolunu ekle
                    labels.append(label)  # Etiketi ekle



    # Eğer listeler boşsa veya eşit değilse hata ver
    if not filepaths or not labels:
        raise ValueError("Filepaths or labels are empty.")
    if len(filepaths) != len(labels):
        raise ValueError("Filepaths and labels must have the same length.")

    # Veri çerçevesi oluştur ve döndür
    data = {'Filepath': filepaths, 'Label': labels}
    df = pd.DataFrame(data)
    return df



# CNN modeli oluşturma fonksiyonu
def create_cnn_model(input_shape, num_classes):
    model = Sequential()  # Boş bir model oluştur
    model.add(Conv2D(32, (4, 4), activation='relu',padding="same", input_shape=input_shape))  # İlk konvolüsyon katmanı
    # Conv2D: Girdi görüntüsüne konvolüsyonel filtre uygular, özellikleri çıkarır.
    # Kernel: (5, 5) değerleri, konvolüsyonel katmanın kernel boyutunu belirtir.

    model.add(MaxPooling2D(pool_size=(2, 2)))  # İlk max pooling katmanı
    # MaxPooling2D: Özellik haritasının boyutunu küçültür, en yüksek değerleri alır.
    # Pooling: pool_size=(2, 2), havuzlama penceresinin boyutunu belirtir.

    model.add(Conv2D(64, (4, 4), activation='relu',padding="same"))  # İkinci konvolüsyon katmanı
    model.add(MaxPooling2D(pool_size=(2, 2)))  # İkinci max pooling katmanı

    model.add(Flatten())  # Çok boyutlu veriyi düzleştir
    # Flatten: Konvolüsyonel ve pooling katmanlarından çıkan çok boyutlu veriyi düzleştirir.
    # Flatten Katmanı: Çok boyutlu veriyi tek boyutlu hale getirir.

    model.add(Dense(128, activation='relu'))  # Tam bağlantılı (dense) katman
    # Dense: Girişteki tüm nöronları bir sonraki katmana bağlar.
    # Tam Bağlantılı Katmanlar: Modelin çıkış katmanı da Dense(num_classes, activation='softmax') ile belirtilmiştir.

    model.add(Dropout(0.5))  # Dropout katmanı, aşırı öğrenmeyi önlemek için
    # Dropout: Eğitim sırasında rastgele %50 oranında nöronları devre dışı bırakır.

    model.add(Dense(num_classes, activation='softmax',kernel_initializer=HeNormal()))  # Çıkış katmanı
    # softmax: Her sınıfın olasılığını döndürür, en yüksek olasılığa sahip sınıf seçilir.
    # Aktivasyon Fonksiyonu: activation='softmax' sınıfların olasılıklarını hesaplar.

    return model  # Modeli döndür



# Veri çerçevelerini oluştur
train_df = create_dataframe(train_dir)  # Eğitim verileri çerçevesi
test_df = create_dataframe(test_dir)  # Test verileri çerçevesi
validation_df = create_dataframe(validation_dir)  # Doğrulama verileri çerçevesi

input_shape = (150, 150, 3)  # Giriş görüntü boyutu (150x150 piksel, RGB)
num_classes = len(train_df['Label'].unique())  # Farklı etiket sayısı

# Modeli oluştur ve derle
optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)#: Adam optimizasyon algoritması kullanılır.
"""


Değişim önerileri:

Learning rate (Öğrenme oranı):

Genellikle 0.001 olarak başlamak iyi bir tercih olur.
Modelinizin çok yavaş öğrendiğini gözlemlerseniz, öğrenme oranını artırabilirsiniz (örneğin, 0.01).
Eğer model aşırı sapma (divergence) gösteriyorsa, öğrenme oranını düşürmek faydalı olabilir (0.0001 gibi).
beta_1 ve beta_2:

beta_1: 0.9 en yaygın kullanılan değer ve genellikle değiştirilmez.

beta_2: 0.999 de yaygın bir değerdir, genellikle daha nadir durumlarda değiştirilir. 
Daha küçük değerler, daha hızlı ama daha dalgalı bir optimizasyon süreci sağlayabilir.

epsilon:Bu parametre genellikle 1e-07 veya 1e-08 civarında kalır. 
Modelde çok büyük veya çok küçük gradyanlar varsa epsilon değerini artırmak faydalı olabilir.
Ortalama kullanım senaryoları:

Basit modeller ve küçük veri setleri: Öğrenme oranı genellikle 0.001 seviyesinde yeterli olur.
Daha karmaşık modeller ve büyük veri setleri: Öğrenme oranını 0.0001 seviyesine çekmek overfitting'i önleyebilir.


"""
cnn_model = create_cnn_model(input_shape, num_classes)
cnn_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# loss='sparse_categorical_crossentropy: Kayıp fonksiyonu olarak sparse categorical crossentropy kullanılır çoklu sınıflar için uygundur.
# metrics=['accuracy']: Doğruluk metriği hesaplanır.


# Model özetini yazdır
cnn_model.summary()


# Veri artırma ve normalleştirme
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Görüntü verisini [0, 1] aralığına dönüştürerek normalleştirir.

    #rotation_range=20,
    # Görüntüleri rastgele 20 derece döndürür, modelin döndürülmüş görüntülerle daha iyi öğrenmesini sağlar.

    #width_shift_range=0.2,
    # Görüntüleri yatayda %20 oranında kaydırır, modelin kaydırılmış görüntülerle eğitilmesini sağlar.

    #height_shift_range=0.2,  # Görüntüleri dikeyde %20 oranında kaydırır.

    shear_range=0.1,  # Görüntüleri rastgele kesme (shear) işlemi uygular, bu da görüntülerin perspektifini değiştirir.

    zoom_range=0.1,  # Görüntüleri rastgele %10 oranında büyütür veya küçültür.

    #horizontal_flip=True,  # Görüntüleri yatayda rastgele çevirir, böylece simetrik görüntülerle daha iyi öğrenir.

    fill_mode='nearest'  # Dönüşüm veya kaydırma işlemi sonrası boş kalan alanları en yakın piksel değeriyle doldurur.
)   #normalizasyon
# Rescale: Görüntü verisini [0, 1] aralığına dönüştürerek normalleştirir.


# Eğitim verilerini yükleme
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,           # Eğitim verileri için veri çerçevesi
    x_col='Filepath',             # Görüntü dosyalarının yollarını tutan sütun
    y_col='Label',                # Etiketlerin tutulduğu sütun
    color_mode='rgb',
    target_size=(150, 150),       # Her görüntünün yeniden boyutlandırılacağı boyut (150x150 piksel)
    class_mode='sparse',          # Etiketlerin sparse formatında alınması, çoklu sınıflar için uygundur
    batch_size=16,                # Her adımda işlenecek görüntü sayısı
    shuffle=True,                 # Görüntüleri karıştırarak yükleyin
    seed = 0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
)
# batch_size: Her bir işlem adımında modelin eğitiminde kullanılacak örnek sayısıdır.





# Doğrulama verileri için normalizasyon
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
# ImageDataGenerator: Görüntüleri gerçek zamanlı olarak artırmak ve normalleştirmek için kullanılır.
# rescale: Görüntü verisini [0, 1] aralığına dönüştürmek için kullanılır. Bu, modelin daha hızlı ve verimli bir şekilde öğrenmesini sağlar.



# Doğrulama verilerini yükleme
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,       # Doğrulama verileri için veri çerçevesi
    x_col='Filepath',              # Görüntü dosyalarının yollarını tutan sütun
    y_col='Label',                 # Etiketlerin tutulduğu sütun
    target_size=(150, 150),        # Her görüntünün yeniden boyutlandırılacağı boyut (150x150 piksel)
    class_mode='sparse',           # Etiketlerin sparse formatında alınması
    batch_size=16                  # Her adımda işlenecek görüntü sayısı
# batch_size: Her bir işlem adımında modelin eğitiminde kullanılacak örnek sayısıdır.
# Avantaj: İyi bir denge ile eğitim süresini kısaltabilir ve modelin genelleme yeteneğini artırabilir.
# Riskler: Çok büyük batch boyutları bellek sorunlarına ve modelin yerel minimumlara takılmasına yol açabilir.
)


# Modeli eğitme
cnn_model.fit(
    train_generator,# Eğitim verileri
    steps_per_epoch=len(train_generator),  # Her epochta kaç adım olacağını belirler.
    # Toplam eğitim örneği sayısını batch boyutuna bölerek hesaplanır.
    epochs=20,      # Eğitim sırasında geçilecek epoch sayısı
    # Modelin kaç kez eğitim verileri üzerinde geçeceğini belirler.
    # Avantaj: Daha fazla epoch, modelin daha fazla öğrenmesine olanak tanır.
    # Riskler: Aşırı epoch, overfittinge yol açabilir. Model eğitim verilerine çok iyi uyum sağlarken yeni verilerde kötü performans gösterebilir.
    validation_data=validation_generator,  # Doğrulama verileri
    # Eğitim sırasında modelin doğruluğunu ve kaybını izlemek için kullanılır.
    validation_steps=len(validation_generator)  # Doğrulama sırasında kaç adım olacağını belirler.
    # Toplam doğrulama örneği sayısını batch boyutuna bölerek hesaplanır.
)

# Modelin performansını test verileri ile değerlendirme
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    zoom_range=0.1,
    fill_mode='nearest'
)  # Test verileri için normalizasyon
# Test verilerini değerlendirmek için kullanılacak veri artırma ve normalizasyon işlemi.

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(150, 150),
    color_mode='rgb',
    class_mode='sparse',
    batch_size=16,
    shuffle=False
)
# batch_size: Her bir işlem adımında modelin eğitiminde kullanılacak örnek sayısıdır.


# Test verileri ile değerlendirme
loss, accuracy = cnn_model.evaluate(test_generator, steps=len(test_generator))  # Test verilerini değerlendir
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
cnn_model.save('my_model.h5')

