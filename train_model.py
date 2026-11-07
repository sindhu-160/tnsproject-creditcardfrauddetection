import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv('creditcard.csv')

# 2. Separate features (X) and target label (y)
X = df.drop(columns=['Class']) # 'Class' column contains 0 (Normal) or 1 (Fraud)
y = df['Class']

# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Initialize and train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 5. Save the trained model and feature names using joblib
joblib.dump(model, "fraud_model.joblib")
joblib.dump(X.columns.tolist(), "feature_names.joblib")

print("Training complete! Model and feature names saved successfully.")