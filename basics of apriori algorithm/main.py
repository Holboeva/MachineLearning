## Import Libraries
import pandas as pd
import re
from mlxtend.frequent_patterns import association_rules, apriori
from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

## Load Dataset
dataset_path = "dummy dataset.csv"
df = pd.read_csv(dataset_path)

# Preprocess text data
df['lower_text'] = df['Text'].apply(lambda x: re.sub(r'\W+', ' ', str(x).lower()))
df.reset_index(inplace=True)

## Data Processing for Association Rule Mining
full_data = pd.DataFrame()
for i in range(len(df)):
    words = df['lower_text'].iloc[i].split()
    temp = pd.DataFrame({'item': words, 'index': i})
    full_data = pd.concat([full_data, temp], axis=0)

# Group by transaction and item
transactions_str = full_data[(full_data['item'] != '')].groupby(['index', 'item'])['item'].count().reset_index(name='Count')

# Pivot table
my_basket = transactions_str.pivot_table(index='index', columns='item', values='Count', aggfunc='sum').fillna(0)

# Encoding function
def encode(x):
    return 0 if x <= 0 else 1

my_basket_sets = my_basket.applymap(encode)

## Association Rule Mining
frequent_items = apriori(my_basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.sort_values('confidence', ascending=False, inplace=True)
rules['antecedents_consequents'] = rules[['antecedents', 'consequents']].apply(
    lambda x: list(x['antecedents']) + list(x['consequents']), axis=1)

# Save rules
with open('association_rules_model.pkl', 'wb') as file:
    pickle.dump(rules, file)

## Creating Dash Web Application
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Load the saved model
with open('association_rules_model.pkl', 'rb') as file:
    saved_rules = pickle.load(file)

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Next Word Prediction using Apriori Algorithm"),
    html.I("Enter your input here:"),
    html.Br(),
    dcc.Input(id="input1", type="text", placeholder="Sentence", style={'marginRight': '10px', 'width': '90%'}),
    html.Br(), html.Br(),
    html.Label("Predicted next words:"),
    html.Div(id="output"),
])

@callback(
    Output("output", "children"),
    Input("input1", "value"),
)
def update_output(input1):
    if not input1:
        return "Please enter a sentence."

    new_antecedents = input1.lower().split()

    # Find matching rules
    predicted_consequents = saved_rules[
        saved_rules['antecedents'].apply(lambda x: set(new_antecedents).issubset(x))
    ]

    if predicted_consequents.empty:
        return "No suggestions found."

    temp1 = predicted_consequents.sort_values('confidence', ascending=False).head(20)

    temp1['Recommendations'] = temp1['antecedents_consequents'].apply(
        lambda x: [i for i in x if i not in new_antecedents]
    )

    try:
        recommendation_list = set(np.concatenate(temp1['Recommendations'].values).ravel())
    except:
        recommendation_list = set()

    if not recommendation_list:
        return "No new word recommendations."
    return f"{', '.join(recommendation_list)}"

if __name__ == "__main__":
    app.run(debug=True)

