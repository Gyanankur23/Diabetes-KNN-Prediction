import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Diabetes Linear Regression Model")

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

st.sidebar.header("Adjust Features")

user_input = []
for i, feature in enumerate(feature_names):
    min_val = float(X[:, i].min())
    max_val = float(X[:, i].max())
    default_val = float(X[:, i].mean())

    val = st.sidebar.slider(
        feature,
        min_val,
        max_val,
        default_val,
        step=(max_val - min_val) / 100
    )
    user_input.append(val)

user_input = np.array(user_input).reshape(1, -1)
prediction = model.predict(user_input)[0]

st.subheader("Real-Time Prediction")
st.write(f"Predicted Diabetes Progression: {prediction:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(y_test, y_pred, color='blue', alpha=0.5)
axs[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    lw=2
)
axs[0].set_title("True vs Predicted Values")
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")

axs[1].scatter(X_test[:, 2], y_pred, color='green', alpha=0.7)
axs[1].scatter(user_input[0][2], prediction, color='red', s=100)
axs[1].set_title("BMI vs Predicted Values")
axs[1].set_xlabel("BMI (Feature 2)")
axs[1].set_ylabel("Predicted Diabetes Progression")
axs[1].grid(True)

plt.tight_layout()
st.pyplot(fig)
