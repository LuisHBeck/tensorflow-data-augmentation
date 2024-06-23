import matplotlib.pyplot as plt


def show_dataset(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(len(images)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"Label: {labels[i]}")
            plt.axis("off")
    plt.show()


def show_image(image, label):
    plt.figure(figsize=(4, 4))  # Adjust figure size as needed
    plt.imshow(image.numpy())
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")
    plt.show()


def visualize_difference(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original.numpy())

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented.numpy())
    plt.show()