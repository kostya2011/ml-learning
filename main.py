import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump
import mlflow.sklearn
import os

#
os.environ["MLFLOW_TRACKING_URI"] = "Change me"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Change me"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Change me"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "Change me"

# AWS AK/SK are required to upload artifacts to S3 Bucket
os.environ["AWS_ACCESS_KEY_ID"] = "Change me"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Change me"


def get_data(url='http://data.insideairbnb.com/spain/catalonia/barcelona/2022-09-10/visualisations/listings.csv'):
    raw_df = pd.read_csv(url)
    return raw_df


def data_preparation(df):
    #Impute 0 instead missing values for reviews_per_month
    df.reviews_per_month.fillna(0, inplace=True)

    #Recode license: Exempt=0, other=1 - The host has a license or not
    df['license_Exempt'] = np.where(df['license'] == "Exempt", 0, 1)

    #Determine how license impact on price
    df.groupby(["license_Exempt"], as_index=False).agg(avg_price = ("price","median"), count = ("price","count"))

    df = df[df.price!=0]
    df.loc[:,"log_price"] = np.log(df.loc[:,"price"])
    df.loc[:,"log_number_of_reviews"] = np.log(df.loc[:,"number_of_reviews"])
    df.loc[:,"log_reviews_per_month"] = np.log(df.loc[:,"reviews_per_month"])

    df['neighbourhood_group_Eixample'] = np.where(df['neighbourhood_group'] == "Eixample", 1, 0)

    df = df[["log_price","latitude", "longitude", "log_number_of_reviews","log_reviews_per_month", "minimum_nights",	"reviews_per_month", "number_of_reviews_ltm" , "calculated_host_listings_count", "license_Exempt", 'neighbourhood_group', 'room_type']]
    df = pd.get_dummies(df, columns=['neighbourhood_group','room_type'])

    df = df.drop(['room_type_Hotel room','latitude','neighbourhood_group_Sarrià-Sant Gervasi', 'longitude', 'room_type_Hotel room', 'room_type_Shared room',
    'neighbourhood_group_Sant Martí','neighbourhood_group_Gràcia','neighbourhood_group_Les Corts','neighbourhood_group_Sants-Montjuïc','neighbourhood_group_Horta-Guinardó',
    'neighbourhood_group_Nou Barris','neighbourhood_group_Ciutat Vella','room_type_Shared room','neighbourhood_group_Sant Andreu', 'log_reviews_per_month', 'log_number_of_reviews' ], axis=1)
    return df


def model_train(df):
    from sklearn.model_selection import train_test_split

    x = df.drop('log_price', axis=1)
    y = df['log_price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    model = LinearRegression()
    model.fit(x_train, y_train)

    from sklearn.model_selection import cross_val_score
    cross_val_score(model, x_train, y_train, cv=10).mean()

    import statsmodels.api as sm

    model_new = sm.OLS(y_train, x_train)
    results = model_new.fit()

    # create a DataFrame of predicted values and residuals
    df["predicted"] = results.predict(x)
    df["residuals"] = results.resid

    # drop predicted, residuals
    df = df.drop(["predicted","residuals"], axis=1)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    x = df.drop('log_price', axis=1)
    y = df['log_price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = GradientBoostingRegressor(random_state=0)
    model.fit(x_train, y_train)

    return model


def save_model(model):
    if dump(model, 'model.joblib'):
        print("Model saved")


if __name__ == "__main__":
    mlflow.sklearn.autolog()

    raw_df = get_data()
    df = data_preparation(raw_df)
    model = model_train(df)
    save_model(model)


