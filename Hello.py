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
from streamlit_gsheets import GSheetsConnection
import joblib


st.set_page_config(
    page_title="IS5451 Group06",
    page_icon="👋",
)

st.write("# IS5451 Group06 Final Project 👋")


# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read()
# Print results.
for row in df.itertuples():
    st.write(f"{row.name} has a :{row.pet}:")


def get_data():
    temperature = np.random.uniform(24.5, 29.0)
    wind_speed = np.random.uniform(5.9, 20.5)
    return temperature, wind_speed


# load the model
model = joblib.load("model.joblib")


# create a function that takes the temperature and wind speed as input and returns the probability of rain
def predict_rain(temperature, wind_speed):
    return model.predict_proba([[temperature, wind_speed]])[0][1]


# st.button("Get Current Data", key=1)
if st.button("Get Current Data", key=1):
    t, w = get_data()
    st.write(f"Temperature: {t:.2f}°C")
    st.write(f"Wind Speed: {w:.2f} km/h")
    prob = predict_rain(t, w)
    st.write(f"Probability of Rain: {prob*100:.2f} %")
