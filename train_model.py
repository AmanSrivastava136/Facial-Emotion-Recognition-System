# ==========================================
# FINAL FER SCRIPT (ALL-IN-ONE)
# ==========================================

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# SETTINGS
# ----------------------------
IMG_SIZE = (96, 96)
np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------
# PATHS (CHANGE IF NEEDED)
# ----------------------------
train_data = '/kaggle/input/fer-4000/dataset 4000-1174/train1/'
test_data = '/kaggle/input/fer-4000/dataset 4000-1174/test1/'

# ----------------------------
# LOAD DATA
# ----------------------------
def dataset_loader(directory):
    images, labels = [], []
    for label in os.listdir(directory):
        path = os.path.join(directory, label)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(('.jpg','.png')):
                    images.append(os.path.join(path, file))
                    labels.append(label)
    return images, labels

train_images, train_labels = dataset_loader(train_data)
test_images, test_labels = dataset_loader(test_data)

train_df = pd.DataFrame({'image': train_images, 'label': train_labels})
test_df = pd.DataFrame({'image': test_images, 'label': test_labels})

train_df = train_df.sample(frac=1).reset_index(drop=True)

# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess(image_list):
    data = []
    for img_path in tqdm(image_list):
        img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
        img = np.array(img) / 255.0
        data.append(img)
    return np.array(data).reshape(-1, 96, 96, 1)

x_train_full = preprocess(train_df['image'])
x_test = preprocess(test_df['image'])

# ----------------------------
# ENCODE LABELS
# ----------------------------
le = LabelEncoder()
y_train_full = to_categorical(le.fit_transform(train_df['label']), 7)
y_test = to_categorical(le.transform(test_df['label']), 7)

# ----------------------------
# SPLIT
# ----------------------------
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.2,
    stratify=np.argmax(y_train_full, axis=1)
)

# ----------------------------
# AUGMENTATION
# ----------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# ----------------------------
# MODEL
# ----------------------------
model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape=(96,96,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(7, activation='softmax'))

# ----------------------------
# COMPILE
# ----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0004),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# CALLBACKS
# ----------------------------
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5)

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=[early_stop, lr_reduce]
)

# ----------------------------
# GRAPHS (VERY IMPORTANT)
# ----------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.legend()

plt.show()

# ----------------------------
# EVALUATION
# ----------------------------
print("\nTest Evaluation:\n")
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy:", acc*100)

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# SAMPLE PREDICTIONS
# ----------------------------
plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(96,96), cmap='gray')
    plt.title(f"P:{le.classes_[y_pred_classes[i]]}")
    plt.axis('off')

plt.suptitle("Sample Predictions")
plt.show()

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("FER_96x96final_Model.h5")
print("Model Saved")
