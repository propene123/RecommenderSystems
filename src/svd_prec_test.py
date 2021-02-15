from collections import defaultdict
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV, KFold


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
precs = []
for trainset, testset in kf.split(data):
    alg.fit(trainset)
    predictions = alg.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3)
    print(f'SPLIT {split}')
    print('*'*25)
    avg_prec = sum(prec for prec in precisions.values()) / len(precisions)
    precs.append(avg_prec)
    print(avg_prec)
    print('*'*25)
    split+=1

full_avg_prec = sum(precs) / len(precs)
print(f'Full avg: {full_avg_prec}')
