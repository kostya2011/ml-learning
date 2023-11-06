import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


pd.options.mode.chained_assignment = None


def get_data(url='http://data.insideairbnb.com/spain/catalonia/barcelona/2022-09-10/visualisations/listings.csv'):
    raw_df = pd.read_csv(url)
    return raw_df


def preprocessing(df=get_data()):
    df.reviews_per_month.fillna(0, inplace=True)

    df['license_Exempt'] = np.where(df['license'] == "Exempt", 0, 1)
    df.groupby(["license_Exempt"], as_index=False).agg(avg_price = ("price","median"), count = ("price","count"))
    df = df[df.price != 0]
    df["log_price"] = df["price"].apply(np.log)
    df["log_number_of_reviews"] = df["number_of_reviews"].apply(np.log)
    df["log_reviews_per_month"] = df["reviews_per_month"].apply(np.log)
    df['neighbourhood_group_Eixample'] = np.where(df['neighbourhood_group'] == "Eixample", 1, 0)

    df = df[["log_price","latitude", "longitude", "log_number_of_reviews", "log_reviews_per_month", "minimum_nights", "reviews_per_month", "number_of_reviews_ltm" , "calculated_host_listings_count", "license_Exempt", 'neighbourhood_group', 'room_type']]
    df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'])

    df = df.drop(['room_type_Hotel room', 'latitude', 'neighbourhood_group_Sarrià-Sant Gervasi', 'longitude', 'room_type_Hotel room', 'room_type_Shared room',
    'neighbourhood_group_Sant Martí', 'neighbourhood_group_Gràcia', 'neighbourhood_group_Les Corts', 'neighbourhood_group_Sants-Montjuïc','neighbourhood_group_Horta-Guinardó',
    'neighbourhood_group_Nou Barris', 'neighbourhood_group_Ciutat Vella', 'room_type_Shared room', 'neighbourhood_group_Sant Andreu', 'log_reviews_per_month', 'log_number_of_reviews' ], axis=1)

    x = df.drop('log_price', axis=1)
    y = df['log_price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test