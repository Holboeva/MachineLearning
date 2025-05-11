import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CSV faylni o'qish
swedish_motor_insurance = pd.read_csv("swedish_motor_insurance.csv")

# Scatter plot chizish
sns.scatterplot(x="n_claims", y="total_payment_sek", data=swedish_motor_insurance)

sns.regplot(x="n_claims",
            y="total_payment_sek",
            data=swedish_motor_insurance,
            ci=None)

plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# Ma'lumotlar
data = {
    "n_claims": [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    "total_payment_sek": [0, 5, 7, 15, 18, 25, 45, 70, 120, 160, 200, 230, 250, 300, 350, 400, 420, 450]
}

# DataFrame yaratish va CSV faylga yozish
df = pd.DataFrame(data)
df.to_csv("swedish_motor_insurance.csv", index=False)



# CSV faylni o'qish
swedish_motor_insurance = pd.read_csv("swedish_motor_insurance.csv")

# Trend chizig'i bilan grafik
sns.regplot(
    x="n_claims",
    y="total_payment_sek",
    data=swedish_motor_insurance,
    ci=None
)


mdl_payment_vs_claims = ols("total_payment_sek ~ n_claims", data=swedish_motor_insurance)
mdl_payment_vs_claims = mdl_payment_vs_claims.fit()
print(mdl_payment_vs_claims.params)

fish = pd.DataFrame({
    "mass_g": [600, 450, 700, 200, 300, 350, 900, 800, 1000, 150, 180],
    "species": ["Pike", "Roach", "Pike", "Perch", "Perch", "Roach", "Pike", "Pike", "Pike", "Roach", "Perch"]
})

sns.displot(
    data=fish,
    x="mass_g",
    col="species",
    col_wrap=2,
    bins=9
)

summary_stats = fish.groupby("species")["mass_g"].mean()
print(summary_stats)

mdl_mass_vs_species = ols("mass_g ~ species", data=fish).fit()
print(mdl_mass_vs_species.params)
plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# 1. Bream baliqlarini ajratib olish
fish = pd.read_csv("fish.csv")  # Agar fish CSV fayl bo‘lsa, shu yerda yuklanadi
bream = fish[fish["species"] == "Bream"]
print(bream.head())

# 2. Vizualizatsiya: uzunlik va massa orasidagi bog‘liqlik
sns.regplot(x="length_cm", y="mass_g", data=bream, ci=None)
plt.show()

# 3. Regressiya modelini qurish
mdl_mass_vs_length = ols("mass_g ~ length_cm", data=bream).fit()
print(mdl_mass_vs_length.params)

# 4. Bashorat qilish uchun uzunlik qiymatlari
explanatory_data = pd.DataFrame({"length_cm": np.arange(20, 41)})

# 5. Model yordamida bashorat qilish
prediction_data = explanatory_data.assign(
    mass_g=mdl_mass_vs_length.predict(explanatory_data)
)
print(prediction_data)

# 6. Vizual ko‘rsatish: haqiqiy va bashorat qilingan nuqtalar
fig = plt.figure()
sns.regplot(x="length_cm", y="mass_g", data=bream, ci=None)
sns.scatterplot(
    x="length_cm", y="mass_g", data=prediction_data, color="red", marker="s"
)
plt.show()

# 7. Kichik bream baliqlarining massasi (uzunligi 10 cm)
little_bream = pd.DataFrame({"length_cm": [10]})
pred_little_bream = little_bream.assign(
    mass_g=mdl_mass_vs_length.predict(little_bream)
)
print(pred_little_bream)