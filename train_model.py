"""
train_model.py
يدرب موديل KNN على نفس الـ pipeline اللي في النوت بوك المُعدَّل
(بعد ما اتشال عمود ca بالكامل)، وبيحفظ الموديل + الـ scaler + أسماء الأعمدة.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1) تحميل الداتا
# ----------------------------
df = pd.read_csv("heart_disease_uci.csv")

# ----------------------------
# 2) تنظيف أساسي
# ----------------------------
df.drop("id", axis=1, inplace=True)
df = df.drop("ca", axis=1)   # <-- تم حذف العمود ده زي النوت بوك المُعدَّل

df["thal"] = df["thal"].fillna("unknown")
df["slope"] = df["slope"].fillna("unknown")

df["trestbps"] = df["trestbps"].replace(0, np.nan)
df["chol"] = df["chol"].replace(0, np.nan)

num_cols_to_impute = ["trestbps", "chol", "thalch", "oldpeak"]
for col in num_cols_to_impute:
    df[col] = df[col].fillna(df[col].median())

cat_cols_to_impute = ["exang", "fbs", "restecg"]
for col in cat_cols_to_impute:
    df[col] = df[col].fillna(df[col].mode()[0])

# ----------------------------
# 3) Outlier Capping (IQR)
# ----------------------------
def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
    return df

for col in ["trestbps", "chol", "oldpeak"]:
    df = cap_outliers_iqr(df, col)

# ----------------------------
# 4) Target
# ----------------------------
df["target"] = (df["num"] > 0).astype(int)
df.drop("num", axis=1, inplace=True)

# ----------------------------
# 5) Encoding
# ----------------------------
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["fbs"] = df["fbs"].astype(int)
df["exang"] = df["exang"].astype(int)

nominal_cols = ["cp", "restecg", "slope", "thal", "dataset"]
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# ----------------------------
# 6) X / y + Split
# ----------------------------
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 7) Scaling (بعد الـ split عشان نمنع data leakage)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 8) تدريب موديل KNN
# ----------------------------
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

# ----------------------------
# 9) حفظ الموديل + الـ scaler + أسماء الأعمدة
# ----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\n✅ تم حفظ model.pkl و scaler.pkl و feature_columns.pkl")
