import pandas as pd

data = {
    'InvoiceNo': ['10001', '10002', '10003', '10004', '10005', '10006', '10007', '10008', '10009', '10010'],
    'StockCode': ['A1', 'A2', 'A3', 'A1', 'A2', 'A4', 'A1', 'A3', 'A2', 'A5'],
    'Description': ['Chair', 'Table', 'Lamp', 'Chair', 'Table', 'Candle', 'Chair', 'Lamp', 'Table', 'Vase'],
    'Quantity': [2, 1, 6, 3, 2, 1, 4, 2, 1, 1],
    'InvoiceDate': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'UnitPrice': [15, 25, 10, 15, 25, 8, 15, 10, 25, 12],
    'CustomerID': [12345, 12346, 12347, 12345, 12346, 12348, 12345, 12347, 12346, 12349],
    'Country': ['France', 'France', 'France', 'United Kingdom', 'United Kingdom', 'Portugal', 'Portugal', 'Portugal', 'Sweden', 'Sweden']
}

df = pd.DataFrame(data)
df.to_excel("Online_Retail.xlsx", index=False)
print("✅ 'Online_Retail.xlsx' fayli yaratildi.")


import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Faylni o'qish
data = pd.read_excel('Online_Retail.xlsx')
data['Description'] = data['Description'].str.strip()

# Tozalash
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype(str)
data = data[~data['InvoiceNo'].str.contains('C')]

# Mamlakat bo‘yicha guruhlash funksiyasi
def get_basket(country):
    return (
        data[data['Country'] == country]
        .groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
    )

basket_France = get_basket("France")
basket_UK = get_basket("United Kingdom")
basket_Por = get_basket("Portugal")
basket_Sweden = get_basket("Sweden")

# Hot encoding funksiyasi
def hot_encode(x):
    return 0 if x <= 0 else 1

# Har bir basketga encoding qilish
for basket_name in ['France', 'UK', 'Por', 'Sweden']:
    basket = eval(f"basket_{basket_name}")
    encoded = basket.applymap(hot_encode)
    exec(f"basket_{basket_name} = encoded")

# Apriori va qoidalarni chiqarish funksiyasi
def generate_rules(basket, country, min_support=0.05):
    print(f"\n📦 {country} uchun assotsiatsiya qoidalari:")
    frq_items = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

generate_rules(basket_France, "France", min_support=0.05)
generate_rules(basket_UK, "United Kingdom", min_support=0.01)
generate_rules(basket_Por, "Portugal", min_support=0.05)
generate_rules(basket_Sweden, "Sweden", min_support=0.05)
