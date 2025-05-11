import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
raw_data = pd.read_excel('../online_retail_II.xlsx',sheet_name='Year 2010-2011')

def prepare_retail(dataframe):
    # preparing dataset
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe
df = prepare_retail(raw_data)

def create_apriori_datastructure(dataframe, id=False):
    if id:
        grouped = germany_df.groupby(
            ['Invoice', 'StockCode'], as_index=False).agg({'Quantity': 'sum'})
        apriori_datastructure = pd.pivot(data=grouped, index='Invoice', columns='StockCode', values='Quantity').fillna(
            0).applymap(lambda x: 1 if x > 0 else 0)
        return apriori_datastructure
    else:
        grouped = germany_df.groupby(
            ['Invoice', 'Description'], as_index=False).agg({'Quantity': 'sum'})
        apriori_datastructure = pd.pivot(data=grouped, index='Invoice', columns='Description', values='Quantity').fillna(
            0).applymap(lambda x: 1 if x > 0 else 0)
        return apriori_datastructure

germany_df = df[df['Country'] == 'Germany']

germany_df.head()

germany_apriori_df = create_apriori_datastructure(germany_df,True)
germany_apriori_df.head() # Invoice-Product matrix (apriori data structure)

def get_rules(apriori_df, min_support=0.01):
    frequent_itemsets = apriori(apriori_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)
    return rules
germany_rules = get_rules(germany_apriori_df)
germany_rules.head()

def get_item_name(dataframe, stock_code):
    if type(stock_code) != list:
        product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
        return product_name
    else:
        product_names = [dataframe[dataframe["StockCode"] == product][["Description"]].values[0].tolist()[0] for product in stock_code]
        return product_names

get_item_name(germany_df,10125)

def get_golden_shot(target_id,dataframe,rules):
    target_product = get_item_name(dataframe,target_id)[0]
    recomended_product_ids = recommend_products(rules, target_id)
    recomended_product_names = get_item_name(dataframe,recommend_products(rules, target_id))
    print(f'Target Product ID (which is in the cart): {target_id}\nProduct Name: {target_product}')
    print(f'Recommended Products: {recomended_product_ids}\nProduct Names: {recomended_product_names}')


def recommend_products(rules_df, product_id, rec_count=5):
    sorted_rules = rules_df.sort_values('lift', ascending=False)
    # we are sorting the rules dataframe by using "lift" metric
    recommended_products = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommended_products.append(
                    list(sorted_rules.iloc[i]["consequents"]))

    recommended_products = list({item for item_list in recommended_products for item in item_list})

    return recommended_products[:rec_count]

# simulating some products like they are in cart
TARGET_PRODUCT_ID_1 = 21987
TARGET_PRODUCT_ID_2 = 23235
TARGET_PRODUCT_ID_3 = 22747

get_item_name(germany_df, [TARGET_PRODUCT_ID_1,TARGET_PRODUCT_ID_2, TARGET_PRODUCT_ID_3])

