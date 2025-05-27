import numpy as np
import lmfit

# Define the model function
def model(x, a, b, c):
    return a * np.exp(-b * x) + c

# Create a Model object from the function
model_obj = lmfit.Model(model)

# Generate some sample data
x_data = np.linspace(0, 10, 50)
true_params = {'a': 5, 'b': 0.5, 'c': 1}
y_data = model(x_data, **true_params) + np.random.normal(0, 0.5, 50)

# Set initial parameter values and constraints correctly
params = model_obj.make_params(a=4, b=0.4, c=0.9)
params['a'].min = 0
params['a'].max = 10
params['b'].min = 0
params['b'].max = 1
params['c'].min = 0
params['c'].max = 2

# Fit the model to the data
result = model_obj.fit(y_data, params, x=x_data)

# Calculate AIC
aic = result.aic

# Print the results
print("AIC:", aic)
print(result.fit_report())