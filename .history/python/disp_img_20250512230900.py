import numpy as np
import matplotlib.pyplot as plt

def display_image_slice(image, slice_index, cmap='gray'):
    """
    Displays a single slice of a 3D image.

    Parameters:
        image (numpy.ndarray): The 3D image array of shape (depth, height, width).
        slice_index (int): The index of the slice to display along the depth axis.
        cmap (str): The colormap to use for displaying the image (default is 'gray').
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3D array (depth, height, width).")
    
    if slice_index < 0 or slice_index >= image.shape[0]:
        raise IndexError(f"slice_index {slice_index} is out of bounds for depth {image.shape[0]}.")

    # Extract the slice
    single_slice = image[slice_index, :, :]

    # Display the slice
    plt.imshow(single_slice, cmap=cmap)
    plt.title(f"Slice {slice_index}")
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example: Load a 3D image (replace this with your actual image loading code)
    image = np.random.rand(100, 256, 256)  # Replace with your actual image data

    # Display a specific slice
    display_image_slice(image, slice_index=50)