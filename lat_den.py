import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Path dataset
train_dir = 'splittt_dataset/train'
val_dir = 'splittt_dataset/val'

# Image generator tanpa augmentasi
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Dataset generator
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model DenseNet169
base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Fine-tuning bisa diaktifkan nanti

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Simpan model terbaik otomatis
checkpoint = ModelCheckpoint('model_batik_densenet.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Training model
history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=[checkpoint]
)

# Save model final (opsional)
model.save('model_batik_densenet169_.h5')
