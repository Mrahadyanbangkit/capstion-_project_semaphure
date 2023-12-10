!gdown --id 1dNHpHBfmvDqIKaF0tEcEP48j3IOFE3Nt -O dataset_semaphore.zip

import zipfile

# Unzip zip file
local_zip = '/content/dataset_semaphore.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()

zip_ref.close()

# !mkdir /content/dataset/train /content/dataset/validation

# import os

# base_dir = ''

# print("Contents of base directory:")
# print(os.listdir(base_dir))

# print("\nContents of train directory:")
# print(os.listdir(f'{base_dir}/train'))

# print("\nContents of validation directory:")
# print(os.listdir(f'{base_dir}/validation'))

from sklearn.model_selection import train_test_split
import os
import shutil

base_dir = '/content/dataset'
train_dir = '/content/train'
validation_dir = '/content/validation'

# Bdirektori train dan validation
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Daftar alfabet A-Z
alphabet = 'abcdefghijklmnopqrstuvwxyz'

for letter in alphabet:
    letter_dir = os.path.join(base_dir, letter)
    train_letter_dir = os.path.join(train_dir, letter)
    validation_letter_dir = os.path.join(validation_dir, letter)

    #  list file untuk setiap huruf
    files = os.listdir(letter_dir)

    # Bagi data jadi train dan validation
    train_files, validation_files = train_test_split(files, test_size=0.2, random_state=42)

    # Buat subfolder untuk tiap huruf di direktori train
    os.makedirs(train_letter_dir, exist_ok=True)
    os.makedirs(validation_letter_dir, exist_ok=True)

    # Pindahin file ke direktori train
    for file in train_files:
        src = os.path.join(letter_dir, file)
        dest = os.path.join(train_letter_dir, file)
        shutil.move(src, dest)

    # Pindahin file ke direktori validation
    for file in validation_files:
        src = os.path.join(letter_dir, file)
        dest = os.path.join(validation_letter_dir, file)
        shutil.move(src, dest)

# Contoh: ngeliat 10 data pertama dari setiap subfolder train dan validation
for letter in alphabet:
    train_letter_dir = os.path.join(train_dir, letter)
    validation_letter_dir = os.path.join(validation_dir, letter)

    # liat 10 data pertama dari setiap subfolder train
    print(f'Train files in {letter}:')
    print(os.listdir(train_letter_dir)[:10])

    # liat 10 data pertama dari setiap subfolder validation
    print(f'Validation files in {letter}:')
    print(os.listdir(validation_letter_dir)[:10])

    print('\n' + '='*40 + '\n')

# Contoh: lihat total gambar dari setiap subfolder train dan validation
for letter in alphabet:
    train_letter_dir = os.path.join(train_dir, letter)
    validation_letter_dir = os.path.join(validation_dir, letter)

    # hitung total gambar dari setiap subfolder train
    total_train_files = len(os.listdir(train_letter_dir))

    #hitung total gambar dari setiap subfolder validation
    total_validation_files = len(os.listdir(validation_letter_dir))

    # Menampilkan hasil
    print(f'Total train files in {letter}: {total_train_files}')
    print(f'Total validation files in {letter}: {total_validation_files}')

    print('\n' + '='*40 + '\n')

"""## Coba Build Model

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') > 0.96:
          print("\nakurasi sudah diatas 96%")
          self.model.stop_training = True
callbacks = myCallback()
model.summary()

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])

# Dari lab coursera
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# TRAINING_DIR = "/content/train"
# training_datagen = ImageDataGenerator(
#       rescale = 1./255,
# 	    rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest')

# VALIDATION_DIR = "/content/validation"
# validation_datagen = ImageDataGenerator(rescale = 1./255)

# train_generator = training_datagen.flow_from_directory(
# 	TRAINING_DIR,
# 	target_size=(150,150),
# 	class_mode='categorical',
#   batch_size=126
# )

# validation_generator = validation_datagen.flow_from_directory(
# 	VALIDATION_DIR,
# 	target_size=(150,150),
# 	class_mode='categorical',
#   batch_size=126
# )

#Ngarang sendiiri
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#  direktori train dan validation
train_dir = '/content/train'
validation_dir = '/content/validation'

# Pra-pemrosesan data dan augmentasi gambar
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


# Persiapkan generator untuk data train dan validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # karena multi class
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

history = model.fit(train_generator, epochs=25, validation_data=validation_generator,callbacks=[callbacks])

# dari lab coursera
# history = model.fit(train_generator, epochs=5, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.evaluate(validation_generator)
