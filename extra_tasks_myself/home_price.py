import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Uylarning maydoni (kvadrat metr)
X = np.array([30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)

# Ularning narxi (ming $)
y = np.array([50, 65, 78, 90, 105, 120, 135, 150])

# Model yaratamiz
model = LinearRegression()

# Modelni ma'lumotlarga o'rgatamiz
model.fit(X, y)

# 85 m² uy narxini bashorat qilish
new_house = np.array([[85]])
predicted_price = model.predict(new_house)

print(f"85 m² uy narxi taxminan: {predicted_price[0]} ming $")

# Grafik chizish
plt.scatter(X, y, color='blue', label='Haqiqiy ma’lumotlar')
plt.plot(X, model.predict(X), color='red', linestyle='dashed', label='ML Model')
plt.scatter(new_house, predicted_price, color='green', marker='o', label='Bashorat qilingan narx')

plt.xlabel("Uy maydoni (m²)")
plt.ylabel("Narx (ming $)")
plt.legend()
plt.show()
