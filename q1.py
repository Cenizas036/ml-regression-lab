# Question:
# Implement Multiple Linear Regression to predict student GPA
# using StudyHours, AttendanceRate, FreeTime and GoOut.
# Use train.csv for training and test.csv for testing.
# Evaluate using Mean Squared Error (MSE) and R2 Score.
# Create a scatter plot comparing actual vs predicted GPA.


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



X_train = train[['StudyHours','AttendanceRate','FreeTime','GoOut']]
y_train = train['GPA']

X_test = test[['StudyHours','AttendanceRate','FreeTime','GoOut']]
y_test = test['GPA']



model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)



plt.scatter(y_test, y_pred)
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA")
plt.show()