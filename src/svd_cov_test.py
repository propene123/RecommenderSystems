from collections import defaultdict
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV, KFold


def coverage(predictions, k=5, i_tot=0):
    cov_items = set()
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        user_est_true[uid].append((est, iid))

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        cov_items.update(x[1] for x in user_ratings[:k])

    return len(cov_items) / i_tot


reviews = pd.read_csv('../data/pruned_revs.csv')
pivot = reviews[['user_id', 'business_id', 'stars']]

r = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(pivot, r)

# train, test = train_test_split(data, test_size=0.25)
params = {'n_epochs': 22, 'lr_all': 0.01, 'reg_all': 0.09}
alg = SVD(**params)
# alg.fit(train)
# pred = alg.test(test)
# accuracy.rmse(pred)
# cross_validate(alg, data, measures=['RMSE'], cv=5, verbose=True)
kf = KFold(n_splits=5)
split = 1
covs = []
tot = pivot['business_id'].nunique()
for trainset, testset in kf.split(data):
    alg.fit(trainset)
    predictions = alg.test(testset)
    cov = coverage(predictions, k=5, i_tot=tot)
    print(f'SPLIT {split}')
    print('*'*25)
    covs.append(cov)
    print(cov)
    print('*'*25)
    split += 1

avg_cov = sum(covs) / len(covs)
print(f'Full cov: {avg_cov}')
