import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
x = np.array([2, 4, 6, 8]).reshape(-1, 1)
y = np.array([3, 7, 5, 10])

# Create and fit the model
model = LinearRegression()
model.fit(x, y)

# Get the slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_

# Predict the values
y_pred = model.predict(x)

# Print results
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Linear Equation: y = {m}x + {b}")

# Plotting the results
plt.scatter(x, y, color="blue", label="Original Data")
plt.plot(x, y_pred, color="red", label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression: Mathematical vs Python")
plt.legend()
plt.show()
