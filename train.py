"""
train.py — Train CNN Model for Shelf Stock-Out Detection
Uses Transfer Learning with MobileNetV2 (fast + accurate)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil, requests, zipfile

IMG_SIZE  = (224, 224)
BATCH     = 32
EPOCHS    = 15
MODEL_OUT = "model/shelf_model.h5"


# ── DOWNLOAD FREE DATASET ────────────────────────
def get_dataset():
    """
    FREE Datasets for this project:
    
    Option 1 — SKU110K (best for shelves):
      https://github.com/eg4000/SKU110K_CVPR19
    
    Option 2 — Roboflow Shelf Dataset (easiest):
      https://universe.roboflow.com/
      Search: "empty shelf detection"
      Download → Format: Classification → Use in code
    
    Option 3 — Collect your own! (2 hours work):
      → Go to any grocery store
      → Take 100 photos of full shelves → save in data/stocked/
      → Take 100 photos of empty shelves → save in data/empty/
      → 200 images is enough for 90%+ accuracy!
    
    This script assumes you have:
      data/
        stocked/  ← full shelf images
        empty/    ← empty shelf images
    """
    if not os.path.exists("data/stocked") or not os.path.exists("data/empty"):
        print("📁 Creating sample data folders...")
        os.makedirs("data/stocked", exist_ok=True)
        os.makedirs("data/empty", exist_ok=True)
        print("✅ Folders created!")
        print("\n👉 Add your shelf images to:")
        print("   data/stocked/  → photos of full shelves")
        print("   data/empty/    → photos of empty shelves")
        print("\n   Minimum: 50 images per class")
        return False
    return True


# ── BUILD MODEL ──────────────────────────────────
def build_model():
    """
    Transfer Learning with MobileNetV2
    WHY MobileNetV2?
      → Pre-trained on 1.4M images (ImageNet)
      → Fast inference — works on Raspberry Pi / edge devices
      → 90%+ accuracy with just 100-200 shelf images
      → Small model size (14MB)
    """
    print("🏗️  Building MobileNetV2 model...")
    
    # Load pretrained base
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,       # Remove ImageNet classifier
        weights='imagenet'       # Use pretrained weights
    )
    base.trainable = False       # Freeze base (transfer learning)
    
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # 0=empty, 1=stocked
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"✅ Model built! Parameters: {model.count_params():,}")
    return model


# ── DATA PIPELINE ────────────────────────────────
def build_data_pipeline():
    """
    Data Augmentation — artificially increase dataset size
    Simulates different store lighting, angles, distances
    """
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.7, 1.3],   # Simulate different store lighting
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=0.2
    )
    
    val_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_data = train_gen.flow_from_directory(
        'data/',
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='binary',
        subset='training'
    )
    
    val_data = val_gen.flow_from_directory(
        'data/',
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='binary',
        subset='validation'
    )
    
    return train_data, val_data


# ── FINE TUNING ──────────────────────────────────
def fine_tune(model, train_data, val_data):
    """
    Unfreeze top layers of base model for fine-tuning
    This squeezes extra 2-5% accuracy
    """
    print("\n🔧 Fine-tuning top layers...")
    base = model.layers[0]
    base.trainable = True
    
    # Freeze all except last 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(train_data, validation_data=val_data, epochs=5, verbose=1)
    return model


# ── TRAIN ────────────────────────────────────────
def train():
    print("\n" + "="*50)
    print("  SHELF DETECTOR — MODEL TRAINING")
    print("="*50 + "\n")
    
    if not get_dataset():
        return
    
    train_data, val_data = build_data_pipeline()
    print(f"\n📊 Training samples: {train_data.samples}")
    print(f"📊 Validation samples: {val_data.samples}\n")
    
    model = build_model()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT, save_best_only=True, monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, verbose=1
        )
    ]
    
    print("🚀 Training started...\n")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tune
    model = fine_tune(model, train_data, val_data)
    
    # Final save
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_OUT)
    
    # Results
    val_acc = max(history.history['val_accuracy']) * 100
    print(f"\n{'='*50}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Best Validation Accuracy: {val_acc:.1f}%")
    print(f"   Model saved: {MODEL_OUT}")
    print(f"{'='*50}\n")
    print("▶ Now run: python main.py")


if __name__ == "__main__":
    train()
