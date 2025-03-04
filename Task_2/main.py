import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 10000  # Number of rides
locations = ['Location_A', 'Location_B', 'Location_C', 'Location_D', 'Location_E']
car_types = ['bus', 'minibus', 'van']
payment_methods = ['cash', 'mobile_payment', 'card']
start_date = datetime(2024, 1, 1)

# Generate data
ride_ids = np.arange(1, n_samples + 1)
travel_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
travel_times = [datetime(2024, 1, 1, np.random.randint(0, 24), np.random.randint(0, 60)).time() for _ in range(n_samples)]
travel_from = np.random.choice(locations, n_samples)
travel_to = np.random.choice(locations, n_samples)
car_type = np.random.choice(car_types, n_samples)
max_capacity = np.random.choice([14, 30, 50], n_samples)
payment_method = np.random.choice(payment_methods, n_samples)

# Calculate seat_number based on some logic
# Example logic: Bus type, capacity, time of day, and payment method affect seat_number
seat_number = (np.random.poisson(lam=10, size=n_samples)
               + (max_capacity / 2).astype(int)
               + np.random.randint(0, 5, n_samples)
               - (np.array([t.hour for t in travel_times]) // 3)
               + (payment_method == 'mobile_payment').astype(int) * 5
              ).clip(1, max_capacity)  # Ensure seat_number is between 1 and max_capacity

# Create the DataFrame
data = pd.DataFrame({
    'ride_id': ride_ids,
    'travel_date': travel_dates,
    'travel_time': travel_times,
    'travel_from': travel_from,
    'travel_to': travel_to,
    'car_type': car_type,
    'max_capacity': max_capacity,
    'payment_method': payment_method,
    'seat_number': seat_number
})

data.to_csv("train_revised.csv", index=False)
data.head()


# Feature Engineering
data['travel_date'] = pd.to_datetime(data['travel_date'])
data['day_of_week'] = data['travel_date'].dt.dayofweek
data['month'] = data['travel_date'].dt.month
data['hour'] = data['travel_time'].apply(lambda x: x.hour)

# Drop irrelevant columns
X = data.drop(columns=['ride_id', 'seat_number', 'travel_date', 'travel_time'])
y = data['seat_number']

# Handling Categorical Variables and Scaling
categorical_features = ['payment_method', 'travel_from', 'travel_to', 'car_type']
numerical_features = ['max_capacity', 'day_of_week', 'month', 'hour']

categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

#git remote add origin https://github.com/Holboeva/MachineLearning.git
# git branch -M main
# git push -u origin main