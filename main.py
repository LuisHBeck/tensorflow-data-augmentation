from src.DataAugmentation import DataAugmentation

if __name__ == "__main__":
    dataset_dir = "./assets/dogs"
    augmented_dir = "./datasets/custom/augmented"
    data_augmentation = DataAugmentation(
        dataset_dir=dataset_dir, augmented_dir=augmented_dir
    )

    data_augmentation.get_all_images_paths_and_labels()
    data_augmentation.create_tf_datasets()
    data_augmentation.grayscale_images()