
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the build history dataset
data = pd.read_csv("ml/build_history.csv")

# Convert build status into numeric labels
data["build_status"] = data["build_status"].map({
    "Success": 0,
    "Failure": 1
})

# Separate input features and target
X = data.drop(columns=["build_id", "build_status"])
y = data["build_status"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save the trained model
joblib.dump(model, "ml/build_failure_model.pkl")

print("\nModel saved successfully!")
