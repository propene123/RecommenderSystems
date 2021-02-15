import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

reviews = pd.read_csv('../data/pruned_revs.csv')
pivot = reviews[['user_id', 'business_id', 'stars']]

r = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(pivot, r)

train, test = train_test_split(data, test_size=0.25)
params = {'n_epochs': 22, 'lr_all': 0.01, 'reg_all': 0.09}
alg = SVD(**params)
# alg.fit(train)
# pred = alg.test(test)
# accuracy.rmse(pred)
cross_validate(alg, data, measures=['RMSE'], cv=5, verbose=True)
