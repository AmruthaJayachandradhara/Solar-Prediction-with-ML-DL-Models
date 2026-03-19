import numpy as np, pickle
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

model = tf.keras.models.load_model('best_model.keras')
y_pred = model.predict(X_test).flatten()

# Inverse transform to Watts
inv = lambda y: scalers['target'].inverse_transform(y.reshape(-1,1)).flatten()
y_pred_W = inv(y_pred)
y_test_W = inv(y_test)

mae  = mean_absolute_error(y_test_W, y_pred_W)
rmse = np.sqrt(mean_squared_error(y_test_W, y_pred_W))
r2   = r2_score(y_test_W, y_pred_W)
nrmse = rmse / (y_test_W.max() - y_test_W.min()) * 100

print(f"MAE  = {mae:.2f} W")
print(f"RMSE = {rmse:.2f} W")
print(f"R²   = {r2:.6f}")
print(f"NRMSE = {nrmse:.2f}%")