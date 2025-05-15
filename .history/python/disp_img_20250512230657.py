import numpy as np
import matplotlib.pyplot as plt

# Example: Load a 3D image (replace this with your actual image loading code)
# Assuming the image is a 3D NumPy array of shape (depth, height, width)
image = np.random.rand(100, 256, 256)  # Replace with your actual image data

# Select a single slice (e.g., slice 50 along the depth axis)
slice_index = 50
single_slice = image[slice_index, :, :]

# Display the slice
plt.imshow(single_slice, cmap='gray')
plt.title(f"Slice {slice_index}")
plt.axis('off')
plt.show()