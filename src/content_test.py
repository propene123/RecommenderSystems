import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

bus = pd.read_csv('../data/pruned_bus.csv')

bus = bus[['categories','business_id']]
bus.set_index('business_id', inplace=True)
bus['keywords'] = ''
for index, row in bus.iterrows():
    words=''
    words = words + row['categories']+ ' '
    row['keywords'] = words

bus.drop(columns=['categories'], inplace=True)
count = CountVectorizer()
count_mat = count.fit_transform(bus['keywords'])
sim = cosine_similarity(count_mat, count_mat, dense_output=False)
indices = pd.Series(bus.index)


def recommend(business_id):
    res = []
    idx = indices[indices == business_id].index[0]
    scores = np.array(sim[idx].todense())
    scores = pd.Series(scores[0].flatten().tolist()).sort_values(ascending=False)
    top_10 = list(scores.iloc[1:11].index)
    for i in top_10:
        res.append(list(bus.index)[i])
    return res

kek = bus.index[0]
print(recommend(kek))
