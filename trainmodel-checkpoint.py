import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# Paths
# ----------------------------
train_dir = "split_dataset/train"
val_dir = "split_dataset/val"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

# ----------------------------
# Data Generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ----------------------------
# Compute Class Weights
# ----------------------------
classes = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# ----------------------------
# Build VGG19 Model
# ----------------------------
base_model = VGG19(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ----------------------------
# Compile Model
# ----------------------------
optimizer = SGD(learning_rate=0.0001, momentum=0.9)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# Callbacks
# ----------------------------
if not os.path.exists("model_weights"):
    os.makedirs("model_weights")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ModelCheckpoint(
        "model_weights/cancer_model.keras",
        monitor='val_loss',
        save_best_only=True
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ----------------------------
# Save Final Model
# ----------------------------
model.save("cancer_model_final.keras")

print("Training completed. Model saved.")
