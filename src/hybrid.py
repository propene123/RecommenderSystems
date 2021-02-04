import ast
import numpy as np
import pandas as pd
from surprise import AlgoBase, Dataset, Reader, accuracy
from surprise.prediction_algorithms import SVD
from surprise.model_selection import train_test_split
from surprise import PredictionImpossible, Prediction
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

    def estimate(self, u, i, user_prof, content):
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
        if self.bus_dat.loc[item_id, 'review_count'] < 100:
            if content:
                idx = self.indices[self.indices == item_id].index[0]
                return user_prof.predict(self.count_mat[idx].todense())
            else:
                return self.cfp.estimate(u, i)
        else:
            return self.cfp.estimate(u, i)

    def predict(self, uid, iid, user_prof, content, r_ui=None, clip=True, verbose=False):
        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid, user_prof, content)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred


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
train = data.build_full_trainset()
alg = Hyb(count_mat, indices, reviews, full_bus)
alg.fit(train)
preds = []
user = pivot.iloc[6789, pivot.columns.get_loc('user_id')]
user_revs = reviews.loc[reviews['user_id'] == user]
rated_bus = user_revs.loc[:, 'business_id':'stars']
clf = NearestCentroid()
use_content = False
if user_revs.loc[:, 'stars'].nunique() > 1:
    x = []
    y = []
    for t in rated_bus['business_id']:
        idx = indices[indices == t].index[0]
        counts = np.array(count_mat[idx].todense())
        row = rated_bus[rated_bus['business_id'] == t]['stars'].index
        for j in range(len(row)):
            x.append(counts[0].flatten().tolist())
            y.append(rated_bus[rated_bus['business_id'] == t]['stars'][row[j]])
    x = np.array(x)
    y = np.array(y)
    clf.fit(x, y)
    use_content = True
rated_list = rated_bus['business_id'].values
for bus_id in bus.index:
    if bus_id not in rated_list:
        preds.append(alg.predict(user, bus_id, clf, use_content))

preds.sort(reverse=True, key=lambda x: x.est)
preds = preds[0:5]
for p in preds:
    print(f'ID: {p.iid} || PRED: {p.est}')
