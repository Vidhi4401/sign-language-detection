import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------------
# CONFIG
# -----------------------------
dataset_path = "asl_alphabet_train"
img_size = 128
batch_size = 32
checkpoint_path = "model/checkpoint.keras"
final_model_path = "model/asl_model.keras"
epoch_track_file = "model/epoch.txt"

os.makedirs("model", exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save classes
with open("model/classes.json", "w") as f:
    json.dump(list(train_data.class_indices.keys()), f)

print("Training classes:", train_data.class_indices)

# -----------------------------
# Resume training if checkpoint exists
# -----------------------------
if os.path.exists(checkpoint_path):
    print("\n🔥 Checkpoint found — resuming training...")
    model = load_model(checkpoint_path)
else:
    print("\n🚀 No checkpoint — creating new model...")
    model = Sequential([
        Input(shape=(img_size, img_size, 3)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(train_data.num_classes, activation='softmax')
    ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Load epoch counter
# -----------------------------
if os.path.exists(epoch_track_file):
    with open(epoch_track_file, "r") as f:
        start_epoch = int(f.read().strip())
else:
    start_epoch = 0

print(f"\n➡️ Starting from epoch {start_epoch + 1}")

# -----------------------------
# Save epoch number each epoch
# -----------------------------
class SaveEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global start_epoch
        with open(epoch_track_file, "w") as f:
            f.write(str(start_epoch + epoch + 1))

# -----------------------------
# CHECKPOINT CALLBACK
# -----------------------------
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
epochs_to_train = 1

model.fit(
    train_data,
    validation_data=val_data,
    epochs=start_epoch + epochs_to_train,
    initial_epoch=start_epoch,
    callbacks=[checkpoint, SaveEpochCallback()]
)

model.save(final_model_path)
print("\n🎉 Training complete! Model saved at:", final_model_path)
