from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from data_preprocessing import preprocessing
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


def model_evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    error = mean_absolute_error(predictions, test_labels)
    rmse = mean_squared_error(predictions, test_labels, squared=False)
    """Mean average precision error"""
    mape = 100 * error
    """Model accuracy"""
    accuracy = 100 - mape

    return accuracy, rmse


'''Random Forest Regressor training'''
def rfr_train(x_train, x_test, y_train, y_test):
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='rfr_baseline'):
        params_grid = {
            'bootstrap': [True],
            'max_depth': [90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300],
            'random_state': [1]
        }
        rfr = RandomForestRegressor()

        grid_search = GridSearchCV(estimator=rfr, param_grid=params_grid,
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        grid_accuracy, grid_rmse = model_evaluate(best_model, x_test, y_test)

        mlflow.set_tag("model_name", "RandomForestRegressor")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("RMSE", grid_rmse)
        mlflow.log_metric("Accuracy", grid_accuracy)
        mlflow.sklearn.log_model(
            sk_model=rfr,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-random-forest-regressor-reg-model",
        )

    return best_model


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = preprocessing()
    rfr_train(x_train, x_test, y_train, y_test)



