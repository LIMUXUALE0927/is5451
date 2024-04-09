# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import numpy as np
import datetime
import sqlite3
import joblib


st.set_page_config(
    page_title="IS5451 Group06",
    page_icon="👋",
)

st.write("# IS5451 Group06 Final Project 👋")

conn = sqlite3.connect("weather.db")


def create_table():
    conn.execute(
        "CREATE TABLE IF NOT EXISTS weather (temp REAL, humidity REAL, dt TEXT)"
    )


create_table()


def random_time():
    res = datetime.datetime.now() - datetime.timedelta(
        seconds=np.random.randint(0, 60 * 60 * 24 * 90)
    )
    return res.strftime("%Y-%m-%d %H:%M:%S")


def get_data():
    temp = np.random.uniform(24.8, 30.3)
    humidity = np.random.uniform(65.5, 93.6)
    dt = random_time()
    with conn:
        conn.execute(
            "INSERT INTO weather (temp, humidity, dt) VALUES (?, ?, ?)",
            (temp, humidity, dt),
        )
    return temp, humidity, dt


model = joblib.load("model.joblib")


def predict_rain(temp, humidity):
    prob = model.predict_proba([[temp, humidity]])[0][1]
    return prob


if st.button("Get Current Data", key=1):
    t, h, dt = get_data()
    st.write(f"Temperature: {t:.2f}°C")
    st.write(f"Humidity: {h:.2f}%")
    prob = predict_rain(t, h)
    st.write(f"Probability of Rain: {prob*100:.2f} %")

if st.button("Get Historical Data", key=2):
    data = conn.execute("SELECT * FROM weather")
    # st.table(data) with header
    st.write("temp, humidity, dt")
    st.table(data.fetchall())
