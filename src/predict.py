import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

def predict_ev(battery_capacity, battery_health, mileage, temperature, speed, cycles):

    data = np.array([[battery_capacity, battery_health, mileage, temperature, speed, cycles]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    return prediction