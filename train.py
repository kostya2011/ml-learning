from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from data_preprocessing import preprocessing
import mlflow.sklearn
from mlflow.models import infer_signature
import os

#
os.environ["MLFLOW_TRACKING_URI"] = "Change me"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Change me"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Change me"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "Change me"

# AWS AK/SK are required to upload artifacts to S3 Bucket
os.environ["AWS_ACCESS_KEY_ID"] = "Change me"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Change me"


# Linear Regression training
def lr_train(x_train, x_test, y_train, y_test):
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='lr_baseline'):
        params = {
            "copy_X": True,
            "fit_intercept": True,
            "n_jobs": None,
            "positive": False
        }
        model = LinearRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        signature = infer_signature(x_test, y_pred)

        mlflow.set_tag("model_name", "LinearRegression")
        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name="sk-learn-linear-regression-reg-model",
        )

    return model


# Gradient Boosting Regressor training
def gb_train(x_train, x_test, y_train, y_test):
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='gb_baseline'):
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "squared_error",
        }
        model = GradientBoostingRegressor(**params)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        signature = infer_signature(x_test, y_pred)

        mlflow.set_tag("model_name", "GradientBoosting")
        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name="sk-learn-gradient-boosting-reg-model",
        )

    return model


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = preprocessing()
    lr_train(x_train, x_test, y_train, y_test)
    gb_train(x_train, x_test, y_train, y_test)


