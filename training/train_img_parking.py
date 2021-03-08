import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# show if using CPU or GPU https://www.tensorflow.org/guide/gpu
tf.debugging.set_log_device_placement(True)
# allowing more memory for tensorflow, tensorflow auto use GPU if detected
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# directories
dataset_dir_name = ['parking_dataset']
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dataset_dir_name[0])

# training params
classes_name = ['0', '1']
batch_size_train = 32
img_size = (200, 200)
epochs = 2
AUTOTUNE = tf.data.AUTOTUNE   # automatically set num of parallel worker


def prepare(img):
    # img = tf.image.per_image_standardization(img)  # mean=0; var=1
    rescale = layers.experimental.preprocessing.Rescaling(scale=1./255, input_shape=(img_size[0], img_size[1], 3))
    img = rescale(img)  # mean=0.5, var=1

    return img


# define train and val split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.3,
        subset="training",
        seed=12,
        image_size=img_size,
        batch_size=batch_size_train)

train_ds = train_ds.map(lambda x, y: (prepare(x), y), num_parallel_calls=AUTOTUNE)

# augmentation for training
"""train_ds = train_ds.concatenate(
    train_ds.map(lambda x, y: (tf.image.flip_left_right(x), y), num_parallel_calls=AUTOTUNE).concatenate(
    train_ds.map(lambda x, y: (tf.image.flip_up_down(x), y), num_parallel_calls=AUTOTUNE).concatenate(
    train_ds.map(lambda x, y: (tf.image.flip_left_right(tf.image.flip_up_down(x)), y), num_parallel_calls=AUTOTUNE))))"""

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          dataset_dir,
          validation_split=0.3,
          subset="validation",
          seed=12,
          image_size=img_size,
          batch_size=batch_size_train)

val_ds = val_ds.map(lambda x, y: (prepare(x), y), num_parallel_calls=AUTOTUNE)

"""val_ds = val_ds.concatenate(
    val_ds.map(lambda x, y: (tf.image.flip_left_right(x), y), num_parallel_calls=AUTOTUNE))"""

# configure the dataset for performance
# .cache() keeps the dataset in memory for the next epoch
# .prefect() simultaneous data preprocessing and training process
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# define model and the preprocessing for it
num_classes = 2
model = Sequential([
    keras.Input(shape=(img_size[0], img_size[1], 3)),
    layers.BatchNormalization(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 7, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(1000, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# optimizer
metrics = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(),
              # metrics=['accuracy'])
              metrics=metrics)
model.summary()
weight_for_0 = (1 / 14000) * 37000 / 2.0
weight_for_1 = (1 / 23000) * 37000 / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}

# train the model!
history_plot = model.fit(train_ds,
                         validation_data=val_ds,
                         epochs=epochs,
                         class_weight=class_weight,)

"""# visualize training result
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

lower_acc, upper_acc = 0.95, 1.00
plt.figure(figsize=(8, 8))
ax_acc = plt.subplot(1, 2, 1)
ax_acc.plot(epochs_range, acc, label='Training Accuracy')
ax_acc.plot(epochs_range, val_acc, label='Validation Accuracy')
# ax_acc.set_ylim(lower_acc, upper_acc)
ax_acc.grid(which='both', axis='y')
ax_acc.legend(loc='lower right')
ax_acc.set_title('Training and Validation Accuracy')

lower_loss, upper_loss = 0.00, 0.13
ax_loss = plt.subplot(1, 2, 2)
ax_loss.plot(epochs_range, loss, label='Training Loss');
ax_loss.plot(epochs_range, val_loss, label='Validation Loss')
# ax_loss.set_ylim(lower_loss, upper_loss)
ax_loss.grid(which='both', axis='y')
ax_loss.legend(loc='upper right')
ax_loss.set_title('Training and Validation Loss')
plt.show()"""

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])

        plt.legend()
    plt.show()


plot_metrics(history=history_plot)

model.save_weights('my_checkpoint')

