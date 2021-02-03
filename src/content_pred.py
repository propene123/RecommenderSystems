import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid

bus = pd.read_csv('../data/pruned_bus.csv')
bus = bus[['categories', 'attributes', 'business_id']]
revs = pd.read_csv('../data/pruned_revs.csv')


def get_attr(in_dict):
    res = []
    for key, value in in_dict.items():
        trans = ast.literal_eval(str(value))
        if isinstance(trans, dict):
            res.extend(get_attr(trans))
        else:
            new_word = str(key) + str(value)
            res.append(new_word)
    return res


bus.set_index('business_id', inplace=True)
bus['keywords'] = ''
cols = bus.columns
for index, row in bus.iterrows():
    words = ''
    for c in cols:
        if c == 'attributes':
            if not pd.isna(row[c]):
                words = words + ' '.join(ast.literal_eval(row[c])) + ' '
        else:
            words = words + row['categories'] + ' '
            row['keywords'] = words

bus.drop(columns=['categories', 'attributes'], inplace=True)
count = TfidfVectorizer(analyzer='word', min_df=0)
count_mat = count.fit_transform(bus['keywords'])
indices = pd.Series(bus.index)


def pred(user_id, bus_id):
    user_revs = revs[revs['user_id'] == user_id]
    rated_bus = user_revs[['business_id', 'stars']]
    x = []
    y = []
    for t in rated_bus['business_id']:
        idx = indices[indices == t].index[0]
        counts = np.array(count_mat[idx].todense())
        x.append(counts[0].flatten().tolist())
        row = rated_bus[rated_bus['business_id'] == t]['stars'].index[0]
        y.append(rated_bus[rated_bus['business_id'] == t]['stars'][row])
    x = np.array(x)
    y = np.array(y)
    clf = NearestCentroid()
    clf.fit(x, y)
    idx = indices[indices == bus_id].index[0]
    return clf.predict(count_mat[idx].todense())


print(pred(revs.iloc[4325, revs.columns.get_loc('user_id')], bus.index[3]))
