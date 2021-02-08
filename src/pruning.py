import pandas as pd

covid = pd.read_json('../data/org/covid.json', lines=True)
full_bus = pd.read_csv('../data/pruned_bus.csv')


kek = pd.merge(covid, full_bus, how='inner', on='business_id')
kek = kek.loc[:, ['business_id', 'delivery or takeout', 'Grubhub enabled']]

kek.to_csv('pruned_covid.csv', index=False)
