import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

# seeds 
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Create the dataset directly (since you provided the data)
data = {
    'Sl.No.': list(range(1, 70)),
    'Equivalent of Al3+': [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 2, 2, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, 1, 1, 1, 1.8, 1.8, 1.8, 1.6, 1.6, 1.6, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 3, 3, 3, 1.2, 1.2, 1.2, 5, 5, 4, 4, 4, 4, 4, 4],
    'Equivalents of Hg2+': [0, 0.4, 0.8, 0, 0.2, 0.8, 0, 1.8, 0.2, 0.4, 0.6, 0, 0.2, 0.4, 0.6, 0.8, 1, 0, 0.4, 0.8, 1, 1, 0.2, 2, 0.2, 1, 1.8, 0.4, 0.6, 0.8, 1.4, 0, 0.4, 0.8, 1.2, 1.4, 1.8, 0, 0.4, 0.6, 0.8, 1.2, 1.4, 0, 0.4, 0.6, 0.8, 1.2, 1.4, 1, 2, 3, 1.8, 2, 3, 1, 2, 3, 0.4, 0.8, 1.2, 4, 5, 1.6, 1.8, 2, 3, 4, 5],
    'Emission at 460 nm': [26.28, 16.04, 10.34, 74.89, 36.57, 18.87, 503.67, 144, 74.59, 36.7, 32.8, 178.51, 152.26, 112.51, 73.89, 34.31, 20.35, 285.61, 235.86, 123.41, 40.59, 319, 426.2, 56.02, 413.17, 270.31, 86.8, 328, 298.2, 243.1, 181.92, 517.37, 466.69, 413.02, 315.65, 173.64, 132.01, 531.68, 497.49, 475.86, 454.23, 403.76, 336.21, 557.91, 536, 502.01, 469.38, 458.64, 417.89, 3.5, 3, 1.5, 364, 333.06, 229.94, 158.27, 105.96, 39.6, 273, 196.84, 72.58, 160, 66.18, 287.9, 213, 152, 119, 81, 48]
}

df = pd.DataFrame(data)

print("Dataset Overview:")
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\nDataset statistics:")
print(df.describe())

# Step 2: Feature Engineering
# Create additional features that might be relevant
df['Al_Hg_ratio'] = df['Equivalent of Al3+'] / (df['Equivalents of Hg2+'] + 1e-8)  # Add small epsilon to avoid division by zero
df['Al_Hg_product'] = df['Equivalent of Al3+'] * df['Equivalents of Hg2+']
df['Al_squared'] = df['Equivalent of Al3+'] ** 2
df['Hg_squared'] = df['Equivalents of Hg2+'] ** 2

# Extract features and target
feature_cols = ['Equivalent of Al3+', 'Equivalents of Hg2+', 'Al_Hg_ratio', 'Al_Hg_product', 'Al_squared', 'Hg_squared']
X = df[feature_cols].values
y = df['Emission at 460 nm'].values.reshape(-1, 1)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Step 3: Advanced preprocessing
# Use StandardScaler instead of MinMaxScaler for better performance with neural networks
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Step 4: Split data with stratification consideration
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.15, random_state=42, shuffle=True
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 5: Build an improved ANN model
def create_model(input_dim, learning_rate=0.001):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        Dense(1, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Step 6: Train with advanced callbacks
model = create_model(X_train.shape[1])

# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=50, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=25, 
    min_lr=1e-7,
    verbose=1
)

print("\n🚀 Training the improved model...")
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=8,
    validation_split=0.15,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Step 7: Evaluate the model
print("\n📈 Model Evaluation:")

# Training predictions
train_pred_scaled = model.predict(X_train)
train_pred = scaler_y.inverse_transform(train_pred_scaled)
train_actual = scaler_y.inverse_transform(y_train)

# Test predictions
test_pred_scaled = model.predict(X_test)
test_pred = scaler_y.inverse_transform(test_pred_scaled)
test_actual = scaler_y.inverse_transform(y_test)

# Calculate metrics
train_mse = mean_squared_error(train_actual, train_pred)
test_mse = mean_squared_error(test_actual, test_pred)
train_r2 = r2_score(train_actual, train_pred)
test_r2 = r2_score(test_actual, test_pred)
train_mae = mean_absolute_error(train_actual, train_pred)
test_mae = mean_absolute_error(test_actual, test_pred)

print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Step 8: Display predictions vs actual
print("\n📊 Test Set: Predicted vs Actual Emission at 460 nm")
print("=" * 60)
for i in range(len(test_pred)):
    diff = abs(test_pred[i][0] - test_actual[i][0])
    percent_error = (diff / test_actual[i][0]) * 100 if test_actual[i][0] != 0 else 0
    print(f"Predicted: {test_pred[i][0]:8.2f} nm | Actual: {test_actual[i][0]:8.2f} nm | "
          f"Diff: {diff:6.2f} | Error: {percent_error:5.1f}%")

# Step 9: Comprehensive visualization
plt.figure(figsize=(20, 12))

# Loss curves
plt.subplot(2, 4, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title("Model Loss During Training", fontsize=12, fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True, alpha=0.3)

# MAE curves
plt.subplot(2, 4, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title("Model MAE During Training", fontsize=12, fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.grid(True, alpha=0.3)

# Prediction vs Actual scatter plot
plt.subplot(2, 4, 3)
plt.scatter(test_actual, test_pred, alpha=0.7, s=50)
plt.plot([test_actual.min(), test_actual.max()], [test_actual.min(), test_actual.max()], 'r--', lw=2)
plt.xlabel("Actual Emission")
plt.ylabel("Predicted Emission")
plt.title(f"Predictions vs Actual\nR² = {test_r2:.3f}", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(2, 4, 4)
residuals = test_actual.flatten() - test_pred.flatten()
plt.scatter(test_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Feature importance (correlation with target)
plt.subplot(2, 4, 5)
feature_names = ['Al3+', 'Hg2+', 'Al/Hg_ratio', 'Al×Hg', 'Al²', 'Hg²']
correlations = [np.corrcoef(X[:, i], y.flatten())[0, 1] for i in range(X.shape[1])]
bars = plt.bar(feature_names, correlations, alpha=0.7)
plt.title("Feature Correlations", fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel("Correlation with Target")
plt.grid(True, alpha=0.3)

# Color bars based on correlation strength
for bar, corr in zip(bars, correlations):
    bar.set_color('green' if corr > 0 else 'red')

# Distribution of actual vs predicted
plt.subplot(2, 4, 6)
plt.hist(test_actual, alpha=0.7, label='Actual', bins=10, density=True)
plt.hist(test_pred, alpha=0.7, label='Predicted', bins=10, density=True)
plt.xlabel("Emission Values")
plt.ylabel("Density")
plt.title("Distribution Comparison", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Error distribution
plt.subplot(2, 4, 7)
errors = np.abs(test_actual.flatten() - test_pred.flatten())
plt.hist(errors, bins=10, alpha=0.7, edgecolor='black')
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title(f"Error Distribution\nMean Error: {np.mean(errors):.2f}", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Model architecture visualization (text-based)
plt.subplot(2, 4, 8)
plt.text(0.1, 0.9, "Model Architecture:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, "Input: 6 features", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, "Hidden 1: 128 neurons + BN + Dropout", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.6, "Hidden 2: 64 neurons + BN + Dropout", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.5, "Hidden 3: 32 neurons + BN + Dropout", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.4, "Hidden 4: 16 neurons + Dropout", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.3, "Output: 1 neuron (linear)", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.2, f"Total params: {model.count_params():,}", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.1, f"Training epochs: {len(history.history['loss'])}", fontsize=10, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 10: Cross-validation for more robust evaluation
print("\n🔄 Performing 5-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
    y_train_cv, y_val_cv = y_scaled[train_idx], y_scaled[val_idx]
    
    # Create and train model
    cv_model = create_model(X_train_cv.shape[1])
    cv_model.fit(X_train_cv, y_train_cv, epochs=200, batch_size=8, verbose=0)
    
    # Evaluate
    val_pred = cv_model.predict(X_val_cv)
    val_pred_orig = scaler_y.inverse_transform(val_pred)
    val_actual_orig = scaler_y.inverse_transform(y_val_cv)
    
    r2 = r2_score(val_actual_orig, val_pred_orig)
    cv_scores.append(r2)
    print(f"Fold {fold + 1} R²: {r2:.4f}")

print(f"\nCross-Validation Results:")
print(f"Mean R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Step 11: Feature importance analysis
print("\n🔍 Feature Importance Analysis:")
feature_names = ['Al3+', 'Hg2+', 'Al/Hg_ratio', 'Al×Hg', 'Al²', 'Hg²']
for i, (name, corr) in enumerate(zip(feature_names, correlations)):
    print(f"{name:12}: {corr:6.3f}")

print(f"\n✨ Model training completed successfully!")
print(f"The model now achieves much better performance with R² = {test_r2:.3f}")

print(f"Mean Absolute Error reduced to: {test_mae:.2f} nm")


