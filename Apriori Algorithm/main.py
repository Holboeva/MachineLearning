import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

# Ma'lumotlar
data = {
    'Milk': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Bread': [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    'Butter': [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    'Eggs': [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    'Cheese': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    'Diaper': [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    'Beer': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# 1. Tez-tez uchraydigan itemsetlarni topish
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# 2. Assotsiatsiya qoidalarini qurish
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# 3. Confidence vs Lift diagrammasi
plt.figure(figsize=(8, 6))
plt.scatter(rules['confidence'], rules['lift'], alpha=0.7, color='b')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Confidence vs Lift in Association Rules')
plt.grid()
plt.show()

# 4. Assotsiatsiya qoidalarini network graph ko'rinishida chizish
G = nx.DiGraph()

# Har bir qoida bo‘yicha grafikga qirralar qo‘shish
for _, row in rules.iterrows():
    G.add_edge(', '.join(row['antecedents']), ', '.join(row['consequents']), weight=row['confidence'])

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, k=1.5)  # chiroyli joylashuv
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)

# Qirra ustidagi yozuvlar
edge_labels = {
    (', '.join(row['antecedents']), ', '.join(row['consequents'])): f"{row['confidence']:.2f}"
    for _, row in rules.iterrows()
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Association Rules Network")
plt.show()
