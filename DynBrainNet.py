import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models, constraints

# ============================================================
# Dataset Path
# ============================================================
data_directory = '/kaggle/input/braim-image/Training'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ------------------------------------------------------------
# 1. Load Dataset (train/val split from folder)
# ------------------------------------------------------------
dataset = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=0.2,
    subset="both"   # gives (train, val)
)

train_ds, val_ds = dataset

# Normalize to [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

NUM_CLASSES = len(train_ds.element_spec[1].shape)  # should be 4

# ============================================================
# 2. Custom Components
# ============================================================
class PruneUnderThreshold(constraints.Constraint):
    """Zero out weights below |tau|."""
    def __init__(self, tau=1e-3):
        self.tau = tau
    def __call__(self, w):
        return tf.where(tf.math.abs(w) > self.tau, w, tf.zeros_like(w))
    def get_config(self):
        return {"tau": self.tau}

class LGConv(layers.Layer):
    """Learned Group Convolution (depthwise + grouped pointwise)."""
    def __init__(self, out_channels, kernel_size=3, groups=4, stride=1,
                 tau=1e-3, dropout=0.2, name=None):
        super().__init__(name=name)
        self.dw = layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same",
                                         use_bias=False,
                                         depthwise_constraint=PruneUnderThreshold(tau))
        self.pw = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                                groups=max(1, groups),
                                kernel_constraint=PruneUnderThreshold(tau))
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()
        self.do = layers.Dropout(dropout)
    def call(self, x, training=False):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.do(x, training=training)
        return x

def ResidualBlock(x, out_channels, groups=4, stride=1, tau=1e-3, dropout=0.25, name="res"):
    shortcut = x
    y = LGConv(out_channels, 3, groups, stride, tau, dropout, name=f"{name}_lg1")(x)
    y = LGConv(out_channels, 3, groups, 1, tau, dropout, name=f"{name}_lg2")(y)

    if shortcut.shape[-1] != out_channels or stride != 1:
        shortcut = layers.Conv2D(out_channels, 1, strides=stride, padding="same",
                                 use_bias=False,
                                 kernel_constraint=PruneUnderThreshold(tau))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    out = layers.Add()([shortcut, y])
    out = layers.ReLU()(out)
    return out

# ============================================================
# 3. Build DynBrainNet
# ============================================================
def build_dynbrainnet(input_shape=(224, 224, 3), num_classes=4,
                      groups=4, tau=1e-3, dropout=0.25):
    inp = layers.Input(shape=input_shape)

    # Initial Conv1x1
    x = layers.Conv2D(256, 1, padding="same", use_bias=False,
                      kernel_constraint=PruneUnderThreshold(tau))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 1
    x = ResidualBlock(x, 256, groups, 1, tau, dropout, name="block1")
    x = layers.MaxPooling2D(2)(x)

    # Block 2 + Early Exit 1
    x = ResidualBlock(x, 256, groups, 1, tau, dropout, name="block2")
    ee1 = layers.GlobalAveragePooling2D()(x)
    ee1 = layers.Dropout(0.3)(ee1)
    ee1_out = layers.Dense(num_classes, name="early_exit_1")(ee1)

    x = layers.MaxPooling2D(2)(x)

    # Block 3 + Early Exit 2
    x = ResidualBlock(x, 256, groups, 1, tau, dropout, name="block3")
    ee2 = layers.GlobalAveragePooling2D()(x)
    ee2 = layers.Dropout(0.3)(ee2)
    ee2_out = layers.Dense(num_classes, name="early_exit_2")(ee2)

    # Final Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    main_out = layers.Dense(num_classes, name="main_logits")(x)

    model = models.Model(inputs=inp, outputs=[ee1_out, ee2_out, main_out], name="DynBrainNet")
    return model



# ============================================================
# 4. Compile and Train
# ============================================================
model = build_dynbrainnet(input_shape=IMG_SIZE + (3,), num_classes=4)

# Define losses for each output
losses = {
    "early_exit_1": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    "early_exit_2": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    "main_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
}

# Assign weights to each loss
loss_weights = {
    "early_exit_1": 0.3,
    "early_exit_2": 0.5,
    "main_logits": 1.0
}

# Define metrics for each output
metrics = {k: [tf.keras.metrics.CategoricalAccuracy(name="acc")] for k in losses}

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)
