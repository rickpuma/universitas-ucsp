import mlflow.sklearn
import utils
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
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

logging.info("***** Initiating model registry task *****")


# 1) Set the MLFlow tracking server using the variable "tracking_server_id"

mlflow.set_tracking_uri(tracking_server_id)


# 2) Create an experiment to train and track a Random Forest model 
# including an input example and the signature
experiment_name = "exp_rf_modelregistry"

mlflow.set_experiment(experiment_name)

run_name = "rf_modelregistry"

with mlflow.start_run(run_name=run_name) as run:
    # 3) Train and track a Random Forest model as a Sklearn model
    # including an input example and signature
    model_name = "champion_rf_classifier"
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
     
    input_example = X_train[:5]
    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(clf, model_name, 
                             input_example=input_example,signature=signature)
    run_id = run.info.run_id
    
# 3) Register the trained model in Model Registry
 
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, model_name)

logging.info(f"Model registered as '{model_name}', version: {result.version}")

# 4) List all model versions

client = MlflowClient()
for mv in client.search_model_versions(f"name='{model_name}'"):
    logging.info(f"Version: {mv.version}, state: {mv.current_stage}, run_id: {mv.run_id}")


# 5) Add a tag called "state" to the current model version indicating it is in "staging" using the MLFlow client
client.set_model_version_tag(name=model_name,version=result.version,key="state",value="staging")
logging.info(f"Version {result.version} marked as 'staging' using a tag.")

# 6) Assign the alias "champion" to the current model version using the MLFlow client
alias = "champion"
client.set_registered_model_alias(name=model_name, alias=alias, version=result.version)
logging.info(f"Alias '{alias}' assigned to the version {result.version}.")

# 7) Add a description to the current version of the model using the MLFlow client
description = "RandomForest train on Applications dataset"
client.update_model_version(name=model_name, version=result.version, description=description)
logging.info(f"Description added to the version {result.version}.")

# 8) Load the last registered model version

model_version = "last"
model_uri = f"models:/{model_name}/{model_version}"
registered_model = mlflow.sklearn.load_model(model_uri=model_uri)

logging.info("***** Finalizing model registry task *****")
