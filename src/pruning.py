import pandas as pd

bus = pd.read_csv('../data/pruned_bus.csv')

foo = bus.loc[bus['review_count'] < 10]

bar = len(foo.index)/len(bus.index)
bar *= 100
print(f'{bar}%')
