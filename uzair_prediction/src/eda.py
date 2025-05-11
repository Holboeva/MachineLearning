import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ma'lumotlarni yaratish
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'TotalFlights': [120, 150, 130, 140, 160, 110, 180, 170, 165, 140, 150, 155],
    'TotalDistance': [200000, 230000, 210000, 220000, 240000, 190000, 250000, 240000, 235000, 220000, 225000, 230000],
    'TotalDelay': [5000, 6000, 5500, 5700, 6500, 4800, 7000, 6900, 6750, 6000, 6300, 6400],
    'TotalCanceled': [5, 7, 6, 8, 10, 4, 12, 11, 9, 6, 8, 7]
}

# DataFrame yaratish
df_months = pd.DataFrame(data)

# CSV faylni saqlash
df_months.to_csv('data/12_months_data.csv', index=False)

# Grafiklarni sozlash
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Birinchi grafik: Jami reyslar sonini va masofani chizish
sns.lineplot(data=df_months, x='Month', y='TotalFlights', marker='o', ax=axes[0], label='Total Flights')
sns.lineplot(data=df_months, x='Month', y='TotalDistance', marker='o', ax=axes[0], label='Total Distance')
axes[0].set_title('Total Flights and Distance per Month')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Total (Flights/Distance)')
axes[0].legend()

# Ikkinchi grafik: Kechikish va bekor qilingan reyslarni bar grafikda ko'rsatish
ax2 = axes[1].twinx()
axes[1].bar(df_months['Month'] - 0.2, df_months['TotalDelay'], width=0.4, color='g', alpha=0.6, label='Total Delay')
ax2.bar(df_months['Month'] + 0.2, df_months['TotalCanceled'], width=0.4, color='r', alpha=0.6, label='Total Canceled')

axes[1].set_title('Flight Delays and Cancellations per Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Delay (Minutes)', color='g')
ax2.set_ylabel('Canceled Flights', color='r')
axes[1].legend(loc='upper left')
ax2.legend(loc='upper right')

# Layoutni optimallashtirish va ko'rsatish
plt.tight_layout()
# Grafikni faylga saqlash
plt.savefig('flight_statistics.png')

