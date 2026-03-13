# Question:
# Implement Linear Regression to predict house prices using the
# California Housing dataset (housing.csv).
# Evaluate the model using Mean Squared Error (MSE) and R2 Score.
# Plot actual vs predicted house prices.

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 2: Load dataset
df = pd.read_csv("housing.csv")

# Show first rows
print(df.head())


# Step 3: Select input features
X = df[['total_rooms','population','median_income','housing_median_age']]

# Target variable
y = df['median_house_value']


# Step 4: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Step 5: Create model
model = LinearRegression()


# Step 6: Train model
model.fit(X_train, y_train)


# Step 7: Predict house prices
y_pred = model.predict(X_test)


# Step 8: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# Step 9: Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
