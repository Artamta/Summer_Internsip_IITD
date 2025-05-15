import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

def mask_nifti_sphere(
    input_filepath='test_object.nii',
    output_filepath='test_object_masked.nii',
    grid_size=(110, 110, 10),
    sphere_radius=40
):
    """
    Load a 4D NIfTI file, apply a spherical mask to set voxels outside the sphere to NaN,
    and save the result as a new NIfTI file.
    
    Parameters:
    - input_filepath (str): Path to the input NIfTI file.
    - output_filepath (str): Path where the masked NIfTI file will be saved.
    - grid_size (tuple of ints): Dimensions of the 3D grid (x, y, z).
    - sphere_radius (float): Radius of the central sphere in voxels.
    """
    # Check if the input file exists
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"The input file {input_filepath} does not exist.")
    
    # Load the NIfTI image
    print(f"Loading NIfTI file from: {input_filepath}")
    img = nib.load(input_filepath)
    data = img.get_fdata()
    affine = img.affine
    header = img.header.copy()  # Make a copy to preserve header information
    
    print(f"Original data type: {data.dtype}")
    print(f"Data shape: {data.shape}")
    
    # Validate grid size
    if data.shape[:3] != grid_size:
        raise ValueError(f"Input data grid size {data.shape[:3]} does not match expected {grid_size}.")
    
    # Define the center of the grid
    center = np.array(grid_size) / 2
    print(f"Grid center: {center}")
    print(f"Sphere radius: {sphere_radius}")
    
    # Create a meshgrid for coordinates
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    z = np.arange(grid_size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Compute the distance from the center for each voxel
    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    
    # Create a boolean mask for voxels within the sphere
    sphere_mask = distances <= sphere_radius
    num_voxels_inside = np.sum(sphere_mask)
    num_voxels_outside = np.prod(grid_size) - num_voxels_inside
    print(f"Number of voxels inside the sphere: {num_voxels_inside}")
    print(f"Number of voxels to be set to NaN: {num_voxels_outside}")
    
    # Apply the mask: set voxels outside the sphere to NaN for all b-values
    # The data shape is (110, 110, 10, 13)
    # We need to expand the sphere_mask to 4D to broadcast correctly
    mask_4d = sphere_mask[..., np.newaxis]  # Shape: (110, 110, 10, 1)
    masked_data = np.where(mask_4d, data, np.nan)
    
    # Verify masking
    total_voxels = data.size
    masked_voxels = np.sum(~mask_4d)
    print(f"Total voxels in input data: {total_voxels}")
    print(f"Total voxels masked to NaN: {masked_voxels}")
    
    # Create a new NIfTI image with the masked data
    new_img = nib.Nifti1Image(masked_data, affine, header)
    
    # Save the new NIfTI image
    nib.save(new_img, output_filepath)
    print(f"Masked NIfTI file saved to: {os.path.abspath(output_filepath)}")

def visualize_nifti_slices(file_path, bvalue_idx=0, slice_idx_z=5, slice_idx_y=55, slice_idx_x=55):
    """
    Visualize central slices of a 4D NIfTI file along x, y, and z axes for a specific b-value.
    
    Parameters:
    - file_path (str): Path to the NIfTI file.
    - bvalue_idx (int): Index of the b-value to visualize (0-based).
    - slice_idx_z (int): Slice index along the z-axis.
    - slice_idx_y (int): Slice index along the y-axis.
    - slice_idx_x (int): Slice index along the x-axis.
    """
    # Load the NIfTI image
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Validate bvalue_idx
    if bvalue_idx >= data.shape[3]:
        raise ValueError(f"bvalue_idx {bvalue_idx} is out of bounds for {data.shape[3]} b-values.")
    
    # Extract the specific b-value data
    b_data = data[..., bvalue_idx]
    
    # Define slice indices if not provided
    if slice_idx_z is None:
        slice_idx_z = b_data.shape[2] // 2
    if slice_idx_y is None:
        slice_idx_y = b_data.shape[1] // 2
    if slice_idx_x is None:
        slice_idx_x = b_data.shape[0] // 2
    
    # Extract slices
    slice_z = b_data[:, :, slice_idx_z]
    slice_y = b_data[:, slice_idx_y, :]
    slice_x = b_data[slice_idx_x, :, :]
    
    # Handle NaNs for visualization: use a colormap that can represent NaNs
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')  # Set color for NaNs
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Z-axis slice
    im0 = axes[0].imshow(slice_z.T, origin='lower', cmap=cmap)
    axes[0].set_title(f'B-value {bvalue_idx + 1} - Z Slice {slice_idx_z}')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Y-axis slice
    im1 = axes[1].imshow(slice_y.T, origin='lower', cmap=cmap)
    axes[1].set_title(f'B-value {bvalue_idx + 1} - Y Slice {slice_idx_y}')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # X-axis slice
    im2 = axes[2].imshow(slice_x.T, origin='lower', cmap=cmap)
    axes[2].set_title(f'B-value {bvalue_idx + 1} - X Slice {slice_idx_x}')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define input and output file paths
    input_nifti = "test_object.nii"
    output_nifti = "test_object_masked.nii"
    
    # Define grid size and sphere radius
    grid_size = (110, 110, 10)  # (x, y, z)
    sphere_radius = 40  # in voxels
    
    # Apply the spherical mask
    mask_nifti_sphere(
        input_filepath=input_nifti,
        output_filepath=output_nifti,
        grid_size=grid_size,
        sphere_radius=sphere_radius
    )
    
    # Optional: Visualize slices of the masked NIfTI file
    # Choose a b-value index to visualize (0-based, e.g., 0 for the first b-value)
    bvalue_to_visualize = 0  # Change as needed (0 to 12 for 13 b-values)
    
    # Define slice indices (optional: set to None to auto-center)
    slice_idx_z = 5  # Middle slice along z-axis (for grid size 10, index 5 is out of bounds; adjust accordingly)
    slice_idx_y = 55  # Middle slice along y-axis
    slice_idx_x = 55  # Middle slice along x-axis
    
    # Adjust slice_idx_z based on grid size (0-based indexing)
    # Since grid_size z is 10, valid slice indices are 0 to 9
    slice_idx_z = grid_size[2] // 2  # 5
    
    visualize_nifti_slices(
        file_path=output_nifti,
        bvalue_idx=bvalue_to_visualize,
        slice_idx_z=slice_idx_z,
        slice_idx_y=slice_idx_y,
        slice_idx_x=slice_idx_x
    )
