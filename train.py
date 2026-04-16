import joblib
from dataset import build_dataset
from models import get_model
from sklearn.metrics import accuracy_score, mean_absolute_error

X_train, X_test, y_train, y_test = build_dataset(
    "cv_au_subset/metadata.csv",
    "cv_au_subset/audio",
    feature_set="all",
    task="classification"
)

model = get_model(task="classification", model_type="rf")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")

joblib.dump(model, "age_model.pkl")
