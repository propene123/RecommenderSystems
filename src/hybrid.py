import ast
from datetime import datetime
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
        if self.bus_dat.loc[item_id, 'review_count'] < 50:
            if content:
                idx = self.indices[self.indices == item_id].index[0]
                return user_prof.predict(self.count_mat[idx].todense()), True
            else:
                return self.cfp.estimate(u, i), False
        else:
            return self.cfp.estimate(u, i), False

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
            est, used_content = self.estimate(iuid, iiid, user_prof, content)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False
            details['used_content'] = used_content

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)
            details['used_content'] = False

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


print("Loading data and intialising recommender. Please wait...")

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
users = pd.read_csv('../data/pruned_users.csv')
pivot = reviews[['user_id', 'business_id', 'stars']]
pivot = reviews.loc[:, ['user_id', 'business_id', 'stars']]
print(pivot.head())
r = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(pivot, r)
train = data.build_full_trainset()
alg = Hyb(count_mat, indices, reviews, full_bus)
alg.fit(train)


def get_user_id():
    valid = False
    user_id = None
    while not valid:
        user_id = input("Please enter your user_id:\n")
        if len(users[users['user_id'] == user_id].index) > 0:
            valid = True
        else:
            print("That is not a valid user_id, please try entering another user_id")
    return user_id


def gen_preds(user):
    preds = []
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
                y.append(rated_bus[rated_bus['business_id']
                                   == t]['stars'][row[j]])
        x = np.array(x)
        y = np.array(y)
        clf.fit(x, y)
        use_content = True
    rated_list = rated_bus['business_id'].values
    for bus_id in bus.index:
        if bus_id not in rated_list:
            preds.append(alg.predict(user, bus_id, clf, use_content))
    preds.sort(reverse=True, key=lambda x: x.est)
    if len(preds) >= 5:
        preds = preds[0:5]
    return preds


def inspect_item(num):
    valid = False
    sel = None
    exit_words = ['QUIT', 'quit', 'q', 'Q']
    while not valid:
        sel = input(
            "Please enter the number of the bar you wish to further inspect\nor enter quit/QUIT/q/Q to logout\n")
        try:
            sel = int(sel)
            if 0 < sel <= num:
                valid = True
        except ValueError:
            if sel in exit_words:
                sel = -1
                valid = True
        if not valid:
            print("That is not a vaid response, please try again")
    return sel


def display_item(item, rat, used_content):
    valid = False
    sel = None
    item_row = full_bus.loc[item]
    print('*'*len(item_row['name']))
    print(item_row['name'])
    print('*'*len(item_row['name']))
    rating_str = f'Average Rating: {item_row["stars"]} stars'
    print(rating_str)
    print('*'*len(rating_str))
    print('Address:')
    print(item_row['address'])
    print(item_row['city'])
    print(item_row['state'])
    print(item_row['postal_code'])
    print('*'*len(item_row['address']))
    print('Tags:')
    longest = 0
    cats = item_row['categories'].split(',')
    for c in cats:
        tmp = c.strip()
        longest = max(longest, len(tmp))
        print(c.strip())
    print('*'*longest)
    print("Opening Times:")
    time_dict = ast.literal_eval(item_row['hours'])
    for day, times in time_dict.items():
        tmp = times.split('-')
        ltime = datetime.strptime(tmp[0], '%H:%M')
        rtime = datetime.strptime(tmp[1], '%H:%M')
        ltime = ltime.strftime('%H:%M')
        rtime = rtime.strftime('%H:%M')
        print(f'{day:<13}{ltime}-{rtime}')
    print('*'*24)
    print(f"We think you would rate this estabishment {round(rat)} stars")
    if used_content:
        tmp_str = "Based off of previous establishments you have rated"
        longest = len(tmp_str)
        print(tmp_str)
    else:
        tmp_str = "Based off of what similar users to you have rated it"
        longest = len(tmp_str)
        print(tmp_str)
    print('*'*longest)
    print()
    print("Would you like to:\n1. View amenities for this estabishment\n2. Go back")
    while not valid:
        sel = input()
        if sel == '1':
            sel = 1
            valid = True
        elif sel == '2':
            sel = 2
            valid = True
        else:
            print("That is not a valid response. Please try again")
    return sel


def display_amenities(item):
    for key, value in item.items():
        trans = ast.literal_eval(str(value))
        if isinstance(trans, dict):
            deeper = False
            for val in trans.values():
                if str(val) != 'False':
                    deeper = True
            if deeper:
                print('*'*50)
                print(f'{key}:')
                display_amenities(trans)
        else:
            if str(value) in ('True', 'False'):
                if value:
                    print(f'{key}')
            else:
                print(f'{key}: {trans}')
    print('*'*50)


def show_items(preds):
    while True:
        item_tot = 0
        for i, p in enumerate(preds):
            item_name = full_bus.loc[p.iid, 'name']
            item_tot += 1
            print(f'{i+1}. {item_name}')
        response = inspect_item(item_tot)
        if response != -1:
            while True:
                amen = display_item(
                    preds[response-1].iid, preds[response-1].est, preds[response-1].details['used_content'])
                if amen == 1:
                    amens = preds[response-1].iid
                    print('*'*50)
                    display_amenities(ast.literal_eval(
                        full_bus.loc[amens, 'attributes']))
                    input("Press any key to return to item view...")
                    print('\n')
                elif amen == 2:
                    break
            print('\n')
        else:
            print('\n')
            break


def update_rec(user):
    global pivot
    valid = False
    print("Please enter the id of the bar you would like to rate")
    while not valid:
        the_id = input()
        if the_id in full_bus.index:
            valid = True
            valid_rat = False
            the_name = full_bus.loc[the_id, 'name']
            print(f"Please enter a rating between 1 and 5 stars for {the_name}")
            mask = (pivot['user_id'] == user) & (pivot['business_id'] == the_id)
            if len(pivot.loc[mask].index) > 0:
                print(f'Your current rating for {the_name} is {pivot.loc[mask, "stars"].iloc[0]} stars')
            else:
                print(f'{the_name} is currently unrated by you')
            while not valid_rat:
                rat = input()
                try:
                    rat = int(rat)
                    if 0 < rat < 6:
                        valid_rat = True
                        mask = (pivot['user_id'] == user) & (pivot['business_id'] == the_id)
                        if len(pivot.loc[mask, 'stars'].index) > 0:
                            pivot.loc[mask, 'stars'] = rat
                        else:
                            d = {'user_id': user, 'business_id': the_id, 'stars': rat}
                            pivot = pivot.append(d, ignore_index=True)
                        print(f'Success your rating for {the_name} has been updated to {rat} stars')
                        mask = (pivot['user_id'] == user) & (pivot['business_id'] == the_id)
                    else:
                        valid_rat = False
                        print("That is not a valid response. Please try again...")
                except ValueError:
                    valid_rat = False
                    print("That is not a valid response. Please try again...")
        else:
            valid = False
            print("That bar does not exist in our database. Please try re entering the ID")


def login():
    user = get_user_id()
    user_name = users.loc[users['user_id'] == user, 'name'].iloc[0]
    print(f"Welcome back {user_name}")
    leave = False
    while not leave:
        valid = False
        print("Would you like to:\n1. Generate recommendations\n2. Update a bars rating\n3. Logout")
        while not valid:
            resp = input()
            if resp == '1':
                valid = True
                print("The system is now generating your recommendations in order")
                print("This step can take longer the fewer reviews you have")
                print("\n")
                preds = gen_preds(user)
                show_items(preds)
            elif resp == '2':
                valid = True
                update_rec(user)
            elif resp == '3':
                valid = True
                leave = True
            else:
                valid = False
                print(f'resp={resp}')
                print("That is not a valid response. Please try again")


def start_ui():
    leave = False
    print("\n\n")
    print("Welcome to the Bars4U, the bar recommender system.")
    print("This system will generate a list of 5 bars that we think you may enjoy a visit to")
    print("\n\n")
    while not leave:
        valid = False
        print("Would you like to:\n1. Login\n2. Quit")
        while not valid:
            resp = input()
            if resp == '1':
                valid = True
                login()
            elif resp == '2':
                valid = True
                leave = True
            else:
                valid = False
                print("That is not a valid response. Please try again")


start_ui()
