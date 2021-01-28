import pandas as pd
from surprise import Dataset, Reader, accuracy, KNNWithZScore, SVD
from surprise.model_selection import cross_validate, train_test_split

revs = pd.read_csv('./revs_monke.csv')
pivot = revs[['user_id', 'business_id', 'stars']]


r = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(pivot, r)
cross_validate(KNNWithZScore(), data, measures=['rmse', 'mae'], cv=3, verbose=True)
cross_validate(SVD(), data, measures=['rmse', 'mae'], cv=3, verbose=True)

