import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image

from plt_helper import show_image, show_dataset, visualize_difference

# parameters
batch_size = 8
autotune = tf.data.AUTOTUNE
img_height = 180
img_width = 180
data_dir = '../assets/dogs'

# list all image file paths
all_image_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
all_image_paths.sort()
all_image_labels = np.arange(len(all_image_paths))  # Assign sequential labels

# convert lists to numpy array
all_image_paths = np.array(all_image_paths)
all_image_labels = np.array(all_image_labels)

# split the data into train, validation and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(all_image_paths, all_image_labels, test_size=0.2,
                                                                      random_state=123)
val_paths, test_paths, val_labels, test_labels = train_test_split(test_paths, test_labels, test_size=0.5,
                                                                  random_state=123)


# function to load and preprocess images
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0  # normalize to [0,1]
    return image, label


# create tensorflow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=autotune)
train_ds = train_ds.shuffle(buffer_size=len(train_paths)).batch(batch_size).prefetch(buffer_size=autotune)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=autotune)
val_ds = val_ds.batch(batch_size).prefetch(buffer_size=autotune)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=autotune)
test_ds = test_ds.batch(batch_size).prefetch(buffer_size=autotune)

# Print the first batch of the training dataset
# for images, labels in train_ds.take(1):
#     print(images.shape, labels.shape)
# result = (6, 180, 180, 3)(6, )
# (6 images in the batch, height, width, 3 channels (rgb))

# show_dataset(dataset=train_ds)
image, label = next(iter(train_ds))
# show_image(image[0], label[0])

grayscaled_image = tf.image.rgb_to_grayscale(image[0])
# Convert TensorFlow tensor to numpy array and reshape
grayscaled_image_numpy = grayscaled_image.numpy()
grayscaled_image_numpy = grayscaled_image_numpy.reshape(grayscaled_image_numpy.shape[0],
                                                        grayscaled_image_numpy.shape[1])
# Convert to uint8 (required by PIL)
grayscaled_image_numpy = (grayscaled_image_numpy * 255).astype('uint8')
# Create a PIL Image object
pil_image = Image.fromarray(grayscaled_image_numpy)
# Save the image locally
save_path = f"../datasets/custom/augmented/{label[0]}_grayscaled.jpg"
pil_image.save(save_path)

# visualize_difference(image[0], tf.squeeze(grayscaled_image))

