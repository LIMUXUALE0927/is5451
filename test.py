import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/weather.csv")
df = df[["temp", "humidity", "precip"]]

X = df[["temp", "humidity"]]
y = df["precip"] > 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

t1, t2 = min(df["temp"]), max(df["temp"])
h1, h2 = min(df["humidity"]), max(df["humidity"])


def generate_random_data(t1, t2, h1, h2):
    temp = np.random.uniform(t1, t2)
    humidity = np.random.uniform(h1, h2)
    return temp, humidity


t, w = generate_random_data(t1, t2, h1, h2)
model.predict_proba([[t, w]])[0][1]


joblib.dump(model, "model.joblib")

# load the model
model = joblib.load("model.joblib")


def predict_rain(temp, humidity):
    prob = model.predict_proba([[temp, humidity]])[0][1]
    return prob


predict_rain(t, w)
