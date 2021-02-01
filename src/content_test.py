import ast
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

bus = pd.read_csv('../data/pruned_bus.csv')
bus = bus[['categories', 'attributes', 'business_id']]


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
sim = linear_kernel(count_mat, count_mat, dense_output=False)
indices = pd.Series(bus.index)


def recommend(business_id):
    res = []
    idx = indices[indices == business_id].index[0]
    scores = np.array(sim[idx].todense())
    scores = pd.Series(scores[0].flatten().tolist()
                       ).sort_values(ascending=False)
    top_10 = list(scores.iloc[1:11].index)
    for i in top_10:
        res.append(list(bus.index)[i])
    return res


kek = bus.index[0]
print(recommend(kek))
