import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

# Sample dataset
data = {
    "Hours_Studied": [2,4,6,8,10,5,7,9,3,1],
    "Attendance": [60,70,80,90,95,75,85,92,65,50],
    "Previous_Score": [40,55,65,75,85,60,70,80,50,35],
    "Final_Grade": [45,60,70,80,90,65,75,85,55,40]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied", "Attendance", "Previous_Score"]]
y = df["Final_Grade"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model/grade_model.pkl")
print("Model trained and saved successfully.")
