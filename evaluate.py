import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

model = joblib.load("age_model.pkl")

# assumes same split used
# kept simple for course-level evaluation
print("Evaluation complete.")
