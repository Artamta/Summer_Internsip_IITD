import nibabel as nib
import matplotlib.pyplot as plt

def display_nifti_slice(file_path, slice_index=0):
    """
    Display a specific slice of a NIfTI (.nii.gz) file.

    Parameters:
        file_path (str): Path to the NIfTI file.
        slice_index (int): Index of the slice to display (default is 0).
    """
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Check if the slice index is within bounds
    if slice_index < 0 or slice_index >= data.shape[2]:
        raise ValueError(f"Slice index out of bounds. Must be between 0 and {data.shape[2] - 1}.")

    # Display the slice
    plt.imshow(data[:, :, slice_index], cmap='gray')
    plt.title(f"Slice {slice_index}")
    plt.axis('off')
    plt.show()

# Example usage
file_path = "/Users/ayush/Desktop/project-internsip/Datasets/ivim_chest.nii.gz"
display_nifti_slice(file_path, slice_index=50)  # Change slice_index as needed