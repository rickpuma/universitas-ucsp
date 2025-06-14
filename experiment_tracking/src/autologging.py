import utils
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import mlflow.xgboost
import xgboost as xgb
import logging


# Setting execution logging configuration
logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s {%(pathname)s:%(lineno)d} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
)

# Global variables
tracking_server_id = utils.TRACKING_SERVER_URI


# Loading dataset
X, y = utils.data_loading()
# Preprocessing dataset
X, y = utils.data_preprocessing(X,y)
# Splitting dataset
X_train, X_test, y_train, y_test = utils.data_split(X,y)

logging.info("***** Initiating autologging task *****")

# 1) Set the MLFlow tracking server using the variable "tracking_server_id"

mlflow.set_tracking_uri(tracking_server_id)

# 2) Create an experiment to train and track a Random Forest Regressor model
# as a general model using the MLFlow autologging functionality
experiment_name = "exp_autologging"

mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    mlflow.autolog()
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)    
    rf.fit(X_train, y_train)

# 3) Create an experiment to train and track a Random Forest Regressor model
# as a Sklearn model using the MLFlow autologging functionality
experiment_name = "exp_sklearn_autologging"

mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    mlflow.sklearn.autolog() 
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)    
    rf.fit(X_train, y_train)

# Create an experiment to train and track an XGBoost Regressor model 
# as an XGBoost model using the MLFlow autologging functionality
experiment_name = "exp_xgboost_autologging"

mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    mlflow.xgboost.autolog()
    xgboost = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=3, learning_rate=0.1)
    xgboost.fit(X_train, y_train)

logging.info("***** Finalizing autologging task *****")
