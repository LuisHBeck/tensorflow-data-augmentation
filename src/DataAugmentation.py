import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


@dataclass
class DataAugmentation:
    dataset_dir: str
    augmented_dir: str
    batch_size: int = None
    autotune: int = None
    img_height: int = None
    img_width: int = None

    __train_paths: Any = None
    __test_paths: Any = None
    __val_paths: Any = None

    __train_labels: Any = None
    __test_labels: Any = None
    __val_labels: Any = None

    __train_ds: Any = None
    __test_ds: Any = None
    __val_ds: Any = None

    __image: Any = None
    __label: Any = None

    def __post_init__(self):
        self.batch_size = 8
        self.autotune = tf.data.AUTOTUNE
        self.img_height = 1080
        self.img_width = 1920

    def get_all_images_paths_and_labels(self):
        # list all image file paths
        all_image_paths = [
            os.path.join(self.dataset_dir, file_name)
            for file_name in os.listdir(self.dataset_dir)
        ]
        all_image_paths.sort()
        all_image_labels = np.arange(len(all_image_paths))  # assign sequential labels

        # convert lists to numpy array
        all_image_paths = np.array(all_image_paths)
        all_image_labels = np.array(all_image_labels)

        # split the data into train, validation and test sets
        self.__train_paths, self.__test_paths, self.__train_labels, self.__test_labels = train_test_split(
            all_image_paths, all_image_labels,
            test_size=0.2, random_state=123
        )
        self.__val_paths, self.__test_paths, self.__val_labels, self.__test_labels = train_test_split(
            self.__test_paths, self.__test_labels,
            test_size=0.5, random_state=123
        )

    def __load_and_preprocess_image(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.img_height, self.img_width])
        image = image / 255.0  # normalize to [0,1]
        return image, label

    def __create_augmented_dir(self, augmentation_type: str):
        dir = f"{self.augmented_dir}/{augmentation_type}"
        if not os.path.exists(dir):
            os.makedirs(dir)

    def create_tf_datasets(self):
        self.__train_ds = tf.data.Dataset.from_tensor_slices((self.__train_paths, self.__train_labels))
        self.__train_ds = self.__train_ds.map(
            self.__load_and_preprocess_image, num_parallel_calls=self.autotune
        )
        self.__train_ds = (self.__train_ds.shuffle(
            buffer_size=len(self.__train_paths)
        ).batch(self.batch_size).prefetch(buffer_size=self.autotune))

        self.__val_ds = tf.data.Dataset.from_tensor_slices((self.__val_paths, self.__val_labels))
        self.__val_ds = self.__val_ds.map(
            self.__load_and_preprocess_image, num_parallel_calls=self.autotune
        )
        self.__val_ds = self.__val_ds.batch(self.batch_size).prefetch(buffer_size=self.autotune)

        self.__test_ds = tf.data.Dataset.from_tensor_slices((self.__test_paths, self.__test_labels))
        self.__test_ds = self.__test_ds.map(
            self.__load_and_preprocess_image, num_parallel_calls=self.autotune
        )
        self.__test_ds = self.__test_ds.batch(self.batch_size).prefetch(buffer_size=self.autotune)

    def __save_augmented_img(self, augmented_img, img_label: str, augmentation_type: str):
        # create dir for the augmented images if not exists
        self.__create_augmented_dir(augmentation_type=augmentation_type)

        # convert TensorFlow tensor to numpy array and reshape
        augmented_img_numpy = augmented_img.numpy()

        # check if the image is grayscale or not
        if augmented_img_numpy.ndim == 3 and augmented_img_numpy.shape[2] == 1:
            # grayscale image
            augmented_img_numpy = augmented_img_numpy.reshape(
                augmented_img_numpy.shape[0], augmented_img_numpy.shape[1]
            )
        elif augmented_img_numpy.ndim == 3 and augmented_img_numpy.shape[2] == 3:
            # color image, no need to reshape
            pass
        else:
            raise ValueError(f"Unexpected image shape: {augmented_img_numpy.shape}")

        # convert to uint8 (required by PIL)
        augmented_img_numpy = (augmented_img_numpy * 255).astype('uint8')

        # create a PIL Image object
        pil_image = Image.fromarray(augmented_img_numpy)

        # save the image locally
        save_path = f"{self.augmented_dir}/{augmentation_type}/{img_label}_{augmentation_type}.jpg"
        pil_image.save(save_path)

    def grayscale_images(self):
        images, labels = next(iter(self.__train_ds))
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            grayscaled_image = tf.image.rgb_to_grayscale(image)
            self.__save_augmented_img(
                augmented_img=grayscaled_image,
                img_label=label,
                augmentation_type="grayscale"
            )

    def saturate_images(self):
        images, labels = next(iter(self.__train_ds))
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            saturated_image = tf.image.adjust_saturation(image, 3)
            self.__save_augmented_img(
                augmented_img=saturated_image,
                img_label=label,
                augmentation_type="saturation"
            )

    def bright_images(self):
        images, labels = next(iter(self.__train_ds))
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            bright_image = tf.image.adjust_brightness(image, 0.4)
            self.__save_augmented_img(
                augmented_img=bright_image,
                img_label=label,
                augmentation_type="bright"
            )

    def center_crop_images(self):
        images, labels = next(iter(self.__train_ds))
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            center_cropped_image = tf.image.adjust_brightness(image, 0.4)
            self.__save_augmented_img(
                augmented_img=center_cropped_image,
                img_label=label,
                augmentation_type="center_crop"
            )

    def flip_images(self):
        images, labels = next(iter(self.__train_ds))
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            flipped_image = tf.image.flip_left_right(image)
            self.__save_augmented_img(
                augmented_img=flipped_image,
                img_label=label,
                augmentation_type="flip"
            )
