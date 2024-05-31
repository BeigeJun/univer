import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import struct

# Load MNIST data
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
# Load MNIST data
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
# Open image and label files
fp_image = open('MNIST_data/MNIST/raw/train-images-idx3-ubyte','rb')
fp_label = open('MNIST_data/MNIST/raw/train-labels-idx1-ubyte','rb')

# Read metadata
image_magic_number = fp_image.read(4)
number_of_images = fp_image.read(4)
rows = fp_image.read(4)
cols = fp_image.read(4)

label_magic_number = fp_label.read(4)
number_of_labels = fp_label.read(4)

# Convert metadata to integer
num_images = struct.unpack(">I", number_of_images)[0]
num_rows = struct.unpack(">I", rows)[0]
num_cols = struct.unpack(">I", cols)[0]

# Function to read and display a specific image
def show_mnist_image(index):
    # Calculate the byte position of the image and label
    image_offset = 16 + index * num_rows * num_cols
    label_offset = 8 + index

    # Read the image
    fp_image.seek(image_offset)
    img_bytes = fp_image.read(num_rows * num_cols)
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(num_rows, num_cols)

    # Read the label
    fp_label.seek(label_offset)
    label_byte = fp_label.read(1)
    label = struct.unpack(">B", label_byte)[0]

    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()

    # Display the pixel values as bytes
    print("Pixel values (28x28):")
    for row in img:
        print(" ".join(f"{val:3}" for val in row))
# Example usage: display the 0th image
num=int(input("Byte: "))
show_mnist_image(num)
# Close files
fp_image.close()
fp_label.close()
