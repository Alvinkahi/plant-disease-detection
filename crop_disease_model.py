import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# --------  Path and Parameters --------
DATA_DIR = r"C:\Users\Alvin\Desktop\dataset1\Train"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# --------  Prepare Image Data --------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Store class indices mapping
class_names = list(train_data.class_indices.keys())
print("\nClass Names:", class_names)

# -------- Load MobileNetV2 Base Model --------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------- Train the Model --------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# ------  Evaluate the Model --------
val_loss, val_accuracy = model.evaluate(val_data)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

# -------- Model & Class Names --------
model.save("dataset.h5")
print("Model saved as dataset.h5")

# Save class names to JSON
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved to class_names.json")

# --------  Plot Training History --------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
