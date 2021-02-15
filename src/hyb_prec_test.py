from collections import defaultdict
import ast
import numpy as np
import pandas as pd
from surprise import AlgoBase, Dataset, Reader, accuracy
from surprise.prediction_algorithms import SVD
from surprise.model_selection import train_test_split, cross_validate, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid





def precision_recall_at_k(predictions, k=5, threshold=3):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls




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
                return clf.predict(self.count_mat[idx].todense())[0]
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
alg = Hyb(count_mat, indices, reviews, full_bus)
kf = KFold(n_splits=5)
split = 1
precs = []
for trainset, testset in kf.split(data):
    alg.fit(trainset)
    predictions = alg.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3)
    # Precision and recall can then be averaged over all users
    print(f'SPLIT {split}')
    print('*'*25)
    avg_prec = sum(prec for prec in precisions.values()) / len(precisions)
    precs.append(avg_prec)
    print(avg_prec)
    print('*'*25)
    split+=1

full_avg_prec = sum(precs) / len(precs)
print(f'Full avg: {full_avg_prec}')


