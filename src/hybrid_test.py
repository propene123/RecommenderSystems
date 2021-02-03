import ast
import numpy as np
import pandas as pd
from surprise import AlgoBase, Dataset, Reader, accuracy
from surprise.prediction_algorithms import SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid


def rem_pref(s, pref):
    if s.startswith(pref):
        return s[len(pref):]
    else:
        return s


class Hyb(AlgoBase):

    def __init__(self, tf_idf_mat, bus_idx, rev_dat, bus_dat):
        AlgoBase.__init__(self)
        params = {'n_epochs': 22, 'lr_all': 0.01, 'reg_all': 0.09}
        self.cfp = SVD(**params)
        self.count_mat = tf_idf_mat
        self.indices = bus_idx
        self.rev_dat = rev_dat
        self.bus_dat = bus_dat
        self.bus_dat.set_index('business_id', inplace=True)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.cfp.fit(trainset)
        return self

    def estimate(self, u, i):
        user_id = u
        item_id = i
        try:
            user_id = self.trainset.to_raw_uid(u)
        except ValueError:
            user_id = rem_pref(user_id, 'UKN__')
        try:
            item_id = self.trainset.to_raw_iid(i)
        except ValueError:
            item_id = rem_pref(item_id, 'UKN__')
        if self.bus_dat.loc[item_id, 'review_count'] < 50:
            user_revs = self.rev_dat.loc[self.rev_dat['user_id'] == user_id]
            rated_bus = user_revs.loc[:, 'business_id':'stars']
            if user_revs.loc[:,'stars'].nunique() > 1:
                x = []
                y = []
                for t in rated_bus['business_id']:
                    idx = self.indices[self.indices == t].index[0]
                    counts = np.array(count_mat[idx].todense())
                    row = rated_bus[rated_bus['business_id'] == t]['stars'].index
                    for j in range(len(row)):
                        x.append(counts[0].flatten().tolist())
                        y.append(rated_bus[rated_bus['business_id'] == t]['stars'][row[j]])
                x = np.array(x)
                y = np.array(y)
                clf = NearestCentroid()
                clf.fit(x, y)
                idx = self.indices[self.indices == item_id].index[0]
                return clf.predict(self.count_mat[idx].todense())
            else:
                return self.cfp.estimate(u, i)
        else:
            return self.cfp.estimate(u, i)



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


full_bus = pd.read_csv('../data/pruned_bus.csv')
bus = full_bus[['categories', 'attributes', 'business_id']].copy()
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


reviews = pd.read_csv('../data/pruned_revs.csv')
pivot = reviews[['user_id', 'business_id', 'stars']]
r = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(pivot, r)
train, test = train_test_split(data, test_size=0.25)
# train = data.build_full_trainset()
alg = Hyb(count_mat, indices, reviews, full_bus)
alg.fit(train)
pred = alg.test(test)
# pred = alg.predict(pivot.iloc[123, pivot.columns.get_loc('user_id')], bus.index[3])
accuracy.rmse(pred)
