# 🚘 Car Price Prediction using Machine Learning

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 2️⃣ Load Dataset
df = pd.read_csv("car data.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# 3️⃣ Basic Info
print("\n📘 Dataset Information:")
print(df.info())

# 4️⃣ Check Missing Values
print("\n🧮 Missing Values:\n", df.isnull().sum())

# 5️⃣ Data Overview
print("\n📊 Summary Statistics:\n", df.describe())

# 6️⃣ Data Cleaning & Preprocessing
# Convert categorical columns into numeric using pd.get_dummies
df = pd.get_dummies(df, drop_first=True)

# Independent and dependent variables
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# 7️⃣ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8️⃣ Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9️⃣ Predictions
y_pred = model.predict(X_test)

# 🔟 Model Evaluation
print("\n🎯 Model Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# 1️⃣1️⃣ Feature Importance Visualization
plt.figure(figsize=(8,5))
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh', color='skyblue')
plt.title("Feature Importance for Car Price Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

print("\n✅ Car Price Prediction Analysis Completed Successfully!")
