import matplotlib.pyplot as plt
import cv2

def display_image(image_path, cmap='gray'):
    """
    Displays an image from the given file path.

    Parameters:
        image_path (str): The file path to the image.
        cmap (str): The colormap to use for displaying the image (default is 'gray').
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if cmap == 'gray' else cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Display the image
    plt.imshow(image, cmap=cmap)
    plt.title(f"Image: {image_path}")
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual image file paths
    image_path = "/Users/ayush/Desktop/project-internsip/Datasets/ivim_chest.nii.gz"  # Update this with your image file path
    display_image(image_path)