import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging


# Setting execution logging configuration
logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s {%(pathname)s:%(lineno)d} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
)

# Global variables
DATA_PATH = "data/application_data.csv"
TRACKING_SERVER_URI = "http://localhost:5000"
TEST_SIZE = 0.2
SEED = 42

def data_loading(data_path:str=DATA_PATH) -> tuple[pd.DataFrame,pd.Series]:
    """
    Read input file and separate it into matrix of variables (X) and target (y)
    :param data_path: path of the input data
    :return: Variable matrix and target vector
    """
    logging.info("***** Initiating data loading *****")
    try:
      df = pd.read_csv(DATA_PATH)
    except Exception as e:
        logging.info("Reading error", e)
        return (None,None)

    X = df.drop(["SK_ID_CURR", "TARGET"], axis=1)
    y = df["TARGET"]
    
    logging.info("***** Finalizing data loading *****")

    return X, y

def data_preprocessing(X:pd.DataFrame, y:pd.Series) -> tuple[pd.DataFrame,pd.Series]:
    """
    Encode categorical variables and impute numerical ones
    :param X: Matrix of variables
    :param y: Target vector
    :return: Preprocessed matrix of variables and target vector
    """
    logging.info("***** Initiating data preprocessing *****")

    # Categorical encoding
    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        X[col] = X[col].fillna("missing")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Numerical imputation
    num_cols = X.select_dtypes(include=["number"]).columns
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    logging.info("***** Finalizing data preprocessing *****")

    return X, y

def data_split(X:pd.DataFrame, y:pd.Series,
               test_size:float=TEST_SIZE,
               random_state:int=SEED) -> tuple[pd.DataFrame, pd.DataFrame,
                                           pd.Series, pd.Series]:
    """
    Split matrix variables and target vector into train and test datasets
    :param X: Matrix of variables
    :param y: Target vector
    :param test_size: Size of the test dataset
    :param random_state: Seed for the split
    :return: Matrix of variables and target vectors for train and test
    """
    logging.info("***** Initiating data split *****")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info("***** Finalizing data split *****")

    return X_train, X_test, y_train, y_test