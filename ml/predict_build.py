
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("ml/build_failure_model.pkl")

# Enter details of a new build
new_build = pd.DataFrame([{
    "code_changes": 18,
    "files_changed": 10,
    "tests_run": 28,
    "tests_failed": 3,
    "build_duration": 150,
    "previous_failures": 2
}])

# Predict build status
prediction = model.predict(new_build)[0]
probability = model.predict_proba(new_build)[0][1]

if prediction == 1:
    print("Prediction: BUILD FAILURE")
else:
    print("Prediction: BUILD SUCCESS")

print(f"Failure probability: {probability:.2%}")
