import pandas as pd
import numpy as np

df1 = pd.read_csv("data/DAILYDATA_S24_202312.csv")
df2 = pd.read_csv("data/DAILYDATA_S24_202401.csv")
df3 = pd.read_csv("data/DAILYDATA_S24_202402.csv")
df = pd.concat([df1, df2, df3])

df = df[
    ["Daily Rainfall Total (mm)", "Mean Temperature (°C)", "Mean Wind Speed (km/h)"]
]
df.columns = ["rainfall", "temperature", "wind_speed"]

# train a logistic regression model to predict the probability of rain based on the temperature and wind speed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df[["temperature", "wind_speed"]]
y = df["rainfall"] > 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)

t1, t2 = min(df["temperature"]), max(df["temperature"])
w1, w2 = min(df["wind_speed"]), max(df["wind_speed"])
print(t1, t2)
print(w1, w2)


def generate_random_data(t1, t2, w1, w2):
    temperature = np.random.uniform(t1, t2)
    wind_speed = np.random.uniform(w1, w2)
    return temperature, wind_speed


t, w = generate_random_data(
    min(df["temperature"]),
    max(df["temperature"]),
    min(df["wind_speed"]),
    max(df["wind_speed"]),
)
print(t, w)
model.predict_proba([[t, w]])[0][1]

# save the model
import joblib

joblib.dump(model, "model.joblib")

# load the model
model = joblib.load("model.joblib")
model.predict([[t, w]])


# create a function that takes the temperature and wind speed as input and returns the probability of rain
def predict_rain(temperature, wind_speed):
    return model.predict_proba([[temperature, wind_speed]])[0][1]


predict_rain(t, w)
