import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load
df = pd.read_csv("data/electric_vehicle_analytics.csv")
df.columns = df.columns.str.strip().str.lower()

# Select features
features = [
    'battery_capacity_kwh',
    'battery_health_%',
    'mileage_km',
    'temperature_c',
    'avg_speed_kmh',
    'charge_cycles'
]

X = df[features]
y = df['range_km']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("✅ Model trained successfully!")