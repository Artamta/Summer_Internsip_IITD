# ivim_dki.py

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import psutil
import os
class IVIMDKIAnalysis:
    def __init__(self):
        """
        Initialize class variables to None or default values.
        """
        self.b_values = None
        self.initial_guess = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.learning_rate = None
        self.alpha = None
        self.iterations = None
        self.tv_iterations = None
        self.inp = None
        self.Para = None  # To store parameter maps after analysis
        self.tv_start_iter = None  # Default value

    @staticmethod
    def compute_gradient_magnitudes(data):
        """
        Compute gradient magnitudes of the 3D data using central differences,
        handling NaN values as specified.

        Parameters:
        - data (jax.numpy.DeviceArray): 3D data array.

        Returns:
        - gradient_magnitude (jax.numpy.DeviceArray): Gradient magnitudes.
        """
        # logging.debug("Computing gradient magnitudes.")
        # 
        # Compute gradients using central differences
        grad_x = (jnp.roll(data, -1, axis=0) - jnp.roll(data, 1, axis=0)) / 2
        grad_y = (jnp.roll(data, -1, axis=1) - jnp.roll(data, 1, axis=1)) / 2
        grad_z = (jnp.roll(data, -1, axis=2) - jnp.roll(data, 1, axis=2)) / 2

        # Identify where neighbors are NaN
        nan_x_neg = jnp.isnan(jnp.roll(data, -1, axis=0))
        nan_x_pos = jnp.isnan(jnp.roll(data, 1, axis=0))
        nan_y_neg = jnp.isnan(jnp.roll(data, -1, axis=1))
        nan_y_pos = jnp.isnan(jnp.roll(data, 1, axis=1))
        nan_z_neg = jnp.isnan(jnp.roll(data, -1, axis=2))
        nan_z_pos = jnp.isnan(jnp.roll(data, 1, axis=2))

        # Create masks for valid gradients
        valid_grad_x = ~nan_x_neg & ~nan_x_pos
        valid_grad_y = ~nan_y_neg & ~nan_y_pos
        valid_grad_z = ~nan_z_neg & ~nan_z_pos

        # Apply masks: set gradients to zero where invalid
        grad_x = jnp.where(valid_grad_x, grad_x, 0.0)
        grad_y = jnp.where(valid_grad_y, grad_y, 0.0)
        grad_z = jnp.where(valid_grad_z, grad_z, 0.0)

        # Compute gradient magnitude
        gradient_magnitude = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        # logging.debug("Gradient magnitudes computed.")
        return gradient_magnitude

    def tv_reduction(self, data, alpha, tviterations):
        """
        Apply Total Variation (TV) reduction to the data, handling NaNs appropriately.

        Parameters:
        - data (jax.numpy.DeviceArray): 3D data array.
        - alpha (float): TV regularization parameter.
        - tviterations (int): Number of TV iterations.

        Returns:
        - data (jax.numpy.DeviceArray): TV-reduced data.
        """
        # logging.info(f"Starting TV reduction with alpha={alpha}, iterations={tviterations}.")
        
        for iteration in range(tviterations):
            # Compute gradient magnitudes with NaN handling
            grad_magnitudes = self.compute_gradient_magnitudes(data)
            
            # Update data: subtract alpha * gradient magnitude
            data = data - alpha * grad_magnitudes
            
            # Preserve NaNs in the data
            data = jnp.where(jnp.isnan(data), jnp.nan, data)
            

        return data
    def ivim_dki_loss(self, p, b, y):
    # Ensure b and y are the same length
    min_len = min(len(b), len(y))
    b = b[:min_len]
    y = y[:min_len]
    p = jnp.exp(p)
    D, D_star, f, K = p
    model = y - f * jnp.exp(-b * D_star) - (1 - f) * jnp.exp(-b * D + (b**2 * D**2 * K) / 6)
    loss = jnp.sum(model**2)
    return loss
    
    def ivim_dki_loss(self, p, b, y):
        """
        Compute the loss function for the IVIM-DKI model.

        Parameters:
        - p (jax.numpy.DeviceArray): Log-transformed parameters [ln(D), ln(D_star), ln(f), ln(K)].
        - b (jax.numpy.DeviceArray): B-values.
        - y (jax.numpy.DeviceArray): Measured signal intensities.

        Returns:
        - loss (float): Sum of squared errors.
        """
        p = jnp.exp(p)  # Transform back from log-space
        D, D_star, f, K = p
        model = y - f * jnp.exp(-b * D_star) - (1 - f) * jnp.exp(-b * D + (b**2 * D**2 * K) / 6)
        loss = jnp.sum(model**2)
        return loss

    def adam_step(self, index, param, m, v, b, y, grad_val, beta1=0.9, beta2=0.99, epsilon=1e-8):
        """
        Perform one step of the Adam optimization algorithm.

        Parameters:
        - index (int): Current iteration index.
        - param (jax.numpy.DeviceArray): Current parameters.
        - m (jax.numpy.DeviceArray): First moment vector.
        - v (jax.numpy.DeviceArray): Second moment vector.
        - b (jax.numpy.DeviceArray): B-values.
        - y (jax.numpy.DeviceArray): Measured signal intensities.
        - grad_val (jax.numpy.DeviceArray): Gradient of the loss w.r.t parameters.
        - beta1 (float): Exponential decay rate for the first moment estimates.
        - beta2 (float): Exponential decay rate for the second moment estimates.
        - epsilon (float): Small constant for numerical stability.

        Returns:
        - param (jax.numpy.DeviceArray): Updated parameters.
        - m (jax.numpy.DeviceArray): Updated first moment vector.
        - v (jax.numpy.DeviceArray): Updated second moment vector.
        """
        m = beta1 * m + (1 - beta1) * grad_val
        v = beta2 * v + (1 - beta2) * (grad_val ** 2)
        m_hat = m / (1 - beta1 ** (index + 1))
        v_hat = v / (1 - beta2 ** (index + 1))
        param = param - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        return param, m, v

    def optimize_theta_adam(self, index, y, initial_guess, m, v, b):
        """
        Optimize the parameters using the Adam optimizer.

        Parameters:
        - index (int): Current iteration index.
        - y (jax.numpy.DeviceArray): Measured signal intensities.
        - initial_guess (jax.numpy.DeviceArray): Current parameter guesses.
        - m (jax.numpy.DeviceArray): First moment vector.
        - v (jax.numpy.DeviceArray): Second moment vector.
        - b (jax.numpy.DeviceArray): B-values.

        Returns:
        - updated_param (jax.numpy.DeviceArray): Updated parameters.
        - updated_m (jax.numpy.DeviceArray): Updated first moment vector.
        - updated_v (jax.numpy.DeviceArray): Updated second moment vector.
        """
        guess = jnp.log(initial_guess)  # Work in log-space for positivity constraints
        gradient = grad(self.ivim_dki_loss)(guess, b, y)
        updated_param, updated_m, updated_v = self.adam_step(index, guess, m, v, b, y, gradient)
        updated_param = jnp.exp(updated_param)  # Transform back to original space
        return updated_param, updated_m, updated_v

    def run_ivim_dki_analysis(self):
        """
        Run the IVIM-DKI analysis on the input data.

        Returns:
        - Para (jax.numpy.DeviceArray): Optimized parameter maps with shape (X, Y, Z, 4).
        """
        if self.inp is None or self.b_values is None:
            raise ValueError("Input data and b-values must be set before running the analysis.")

        alpha = self.alpha
        steps = self.iterations
        b = jnp.array(self.b_values)
        inp = self.inp

        # Vectorize the optimize_theta_adam function across the spatial dimensions
        vec_optimise = vmap(self.optimize_theta_adam, in_axes=(None, 0, 0, 0, 0, None))

        # Initialize parameter array and input data
        Para = jnp.full((inp.shape[0], inp.shape[1], inp.shape[2], 4), self.initial_guess)
        dwiData = inp  # Assuming inp has shape (X, Y, slice number, b-value)
        size3d = jnp.prod(jnp.array(dwiData.shape[0:3]))
        ydata_ur = jnp.reshape(dwiData, (size3d, dwiData.shape[3]))

        # Normalize the data by the first time point
        s0 = ydata_ur[:, 0]
        ydata_ur = ydata_ur / (s0[:, jnp.newaxis] + 1e-8)  # Avoid division by zero

        # Handle NaN and Inf values
        ydata_ur = jnp.nan_to_num(ydata_ur, nan=0.0, posinf=0.0, neginf=0.0)

        # Initialize parameter guesses and Adam optimizer states
        initial_guess = jnp.array([0.0013, 0.013, 0.23, 1.1])  # [D, D_star, f, K]
        Para = jnp.tile(initial_guess, (size3d, 1))  # Shape: (size3d, 4)
        m_init = jnp.zeros_like(Para)
        v_init = jnp.zeros_like(Para)
        process = psutil.Process(os.getpid())
        # Optimization loop
        for i in range(steps):
            Para, m_init, v_init = vec_optimise(i, ydata_ur, Para, m_init, v_init, b)
            iter_params = jnp.reshape(Para, (dwiData.shape[0], dwiData.shape[1], dwiData.shape[2], 4))
            if i >= self.tv_start_iter:
                for j in range(4):  # Assuming 4 parameters
                    jnp_3d_map = iter_params[:, :, :, j]
                    jnp_3d_map = jnp.clip(jnp_3d_map, -1e8, 1e8)  # Prevent overflow
                    tv_reduced_map = self.tv_reduction(jnp_3d_map, alpha, self.tv_iterations)
                    iter_params = iter_params.at[:, :, :, j].set(tv_reduced_map)
            iter_params = jnp.nan_to_num(iter_params, nan=0.0, posinf=1e8, neginf=-1e8)
            Para = jnp.reshape(iter_params, (size3d, 4))
            mem_usage_mb = process.memory_info().rss / (1024 * 1024)
            print(f"Iteration {i+1}/{steps}, Memory usage: {mem_usage_mb:.2f} MB")
        # Reshape the final parameters and store in self.Para
        self.Para = jnp.reshape(Para, (dwiData.shape[0], dwiData.shape[1], dwiData.shape[2], 4))
        return self.Para

# PLACEHOLDER CODE!! ####
#########################



# ivim_dki.py

# import numpy as np

# class IVIMDKIAnalysis:
#     def __init__(self):
#         # Initialize class variables with default or dummy values
#         self.b_values = []
#         self.initial_guess = np.array([0.0013, 0.013, 0.23, 1.1])  # Example initial guess
#         self.lower_bounds = np.array([0.0001, 0.001, 0.0, 0.5])      # Example lower bounds
#         self.upper_bounds = np.array([0.005, 0.05, 1.0, 2.0])       # Example upper bounds
#         self.learning_rate = 0.01                                 # Example learning rate
#         self.alpha = 0.1                                          # Example alpha for TV reduction
#         self.iterations = 100                                     # Example number of iterations
#         self.tv_iterations = 10                                   # Example number of TV iterations
#         self.inp = np.zeros((64, 64, 64, 30))                     # Example input data shape (modify as needed)

#     @staticmethod
#     def compute_gradient_magnitudes(data):
#         """
#         Placeholder for computing gradient magnitudes.
#         Returns a zero array of the same shape as input.
#         """
#         return np.zeros_like(data)

#     def tv_reduction(self, data, alpha, tviterations):
#         """
#         Placeholder for total variation reduction.
#         Returns the input data without modification.
#         """
#         return data

#     def ivim_dki_loss(self, p, b, y):
#         """
#         Placeholder for the IVIM-DKI loss function.
#         Returns a dummy loss value (e.g., zero).
#         """
#         return 0.0

#     def adam_step(self, index, param, m, v, b, y, grad, beta1=0.9, beta2=0.99, epsilon=1e-8):
#         """
#         Placeholder for one step of the Adam optimizer.
#         Returns the current parameters without modification.
#         """
#         return param, m, v

#     def optimize_theta_adam(self, index, y, initial_guess, m, v, b):
#         """
#         Placeholder for optimizing parameters using Adam optimizer.
#         Returns the current parameters without modification.
#         """
#         return initial_guess, m, v

#     def run_ivim_dki_analysis(self):
#         """
#         Placeholder for the main IVIM-DKI analysis method.
#         Returns a zeroed parameter array with the expected shape.
#         """
#         alpha = self.alpha
#         steps = self.iterations
#         b = np.array(self.b_values)
#         inp = self.inp

#         # Placeholder shapes (modify based on actual data)
#         dwiData = inp[:, :, :, :]  # Shape: (X, Y, Z, Time)
#         size3d = np.prod(dwiData.shape[0:3])
#         ydata_ur = np.zeros((size3d, dwiData.shape[3]))  # Placeholder flattened data

#         # Normalize the data by the first time point (dummy normalization)
#         s0 = ydata_ur[:, 0]
#         ydata_ur = ydata_ur / (s0[:, np.newaxis] + 1e-8)  # Avoid division by zero

#         # Handle NaN and Inf values (already zeros, but included for completeness)
#         ydata_ur = np.nan_to_num(ydata_ur, nan=0.0, posinf=0.0, neginf=0.0)

#         # Initialize parameter guesses and Adam optimizer states
#         new_shape = (dwiData.shape[0], dwiData.shape[1], dwiData.shape[2], 4)
#         Para = np.full((size3d, 4), self.initial_guess)  # Initial guesses
#         m_init = np.zeros_like(Para)
#         v_init = np.zeros_like(Para)

#         # Optimization loop (placeholder: does not perform actual optimization)
#         for i in range(steps):
#             Para, m_init, v_init = self.optimize_theta_adam(i, ydata_ur, Para, m_init, v_init, b)
#             iter_params = Para.reshape(new_shape)
#             if i >= 40:
#                 for j in range(4):  # Assuming 4 parameters
#                     # Placeholder TV reduction: no change
#                     tv_reduced_map = self.tv_reduction(iter_params[:, :, :, j], self.alpha, self.tv_iterations)
#                     iter_params[:, :, :, j] = tv_reduced_map

#         # Reshape the final parameters and return
#         Para = Para.reshape(new_shape)
#         return Para
