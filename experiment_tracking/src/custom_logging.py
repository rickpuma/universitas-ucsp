import utils
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
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

logging.info("***** Initiating custom logging task *****")


# 1) Set the MLFlow tracking server using the variable "tracking_server_id"

mlflow.set_tracking_uri(tracking_server_id)

# 2) Create an experiment to train and track a Random Forest Regressor model
experiment_name = "exp_rf_customlogging"

mlflow.set_experiment(experiment_name)

run_name = "rf_classifier"
with mlflow.start_run(run_name=run_name):
    # Train model with all the variables
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Feature importance using permutation_importance
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    mask = importances > 0
    filtered_idx = importances.argsort()[::-1][mask[importances.argsort()[::-1]]]
    filtered_features = X_train.columns[filtered_idx]
    X_train_filtered = X_train[filtered_features]
    X_test_filtered = X_test[filtered_features]

    # Train model with selected features
    clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_final.fit(X_train_filtered, y_train)

    # 3) Track the roc_auc, precision, recall and f1-score 
    
    y_pred = clf_final.predict(X_test_filtered)
    y_proba = clf_final.predict_proba(X_test_filtered)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("precision", report["1"]["precision"])
    mlflow.log_metric("recall", report["1"]["recall"])
    mlflow.log_metric("f1-score", report["1"]["f1-score"])
    
    # 4) Graph and track the ROC curve using RocCurveDisplay
    image_file_name = "roc_curve.png"
    image_title = "Receiver Operating Characteristic (ROC) Curve"
    
    roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(image_title)
    plt.savefig(image_file_name)
    plt.close()
    mlflow.log_artifact(image_file_name)
    
    # 5) Graph and save feature importances
    filtered_importances = importances[filtered_idx]
    image_file_name = "permutation_feature_importance.png"
    image_title = "Permutation Feature Importance"
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(filtered_importances)), filtered_importances, align="center")
    plt.xticks(range(len(filtered_importances)), filtered_features, rotation=90)
    plt.tight_layout()
    
    plt.title(image_title)
    plt.savefig(image_file_name)
    plt.close()
    mlflow.log_artifact(image_file_name)
    
    # 6) Log the final model as a Sklearn model including 
    # an input as example
    model_name = "final_model"
    
    input_example = X_test_filtered.iloc[:5]
    mlflow.sklearn.log_model(clf_final, model_name,
                             input_example=input_example)
    
    eval_results = mlflow.evaluate(
        model="runs:/{}/model".format(mlflow.active_run().info.run_id),
        data=X_test_filtered.assign(TARGET=y_test),  # DataFrame with features and target
        targets="TARGET",
        model_type="classifier",
        evaluators="default"
    )
    
    logging.info(f"Results of mlflow.evaluate: {eval_results.metrics}")
    logging.info("***** Finalizing custom logging task *****")