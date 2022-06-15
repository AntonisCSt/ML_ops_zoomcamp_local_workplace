import pandas as pd
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect.task_runners import SequentialTaskRunner
import mlflow
from prefect import flow, task, get_run_logger


@task
def read_data(path):
    df = pd.read_csv(path, delimiter='|')
    return df


@task
def prepare_features(df, categorical, train=True):
    df['dropOff_datetime'] = pd.to_datetime(df['dropOff_datetime'])
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    #saving model

    return

@task
def get_paths(date=None):
    if date == None:
        train_path = '../data/fhv_tripdata_2021-01.csv'
        val_path = '../data/fhv_tripdata_2021-02.csv'
    else:
        year = date.split('-')[0]
        month = int(date.split('-')[1])

        # issue when we have month>10 +and month<2
        train_path = '../data/fhv_tripdata_' + year + '-' + '0' + str(month - 2) + '.csv'
        val_path = '../data/fhv_tripdata_' + year + '-' + '0' + str(month - 1) + '.csv'
    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    joblib.dump(lr, f'./models/model-{date}.bin')
    joblib.dump(dv, f'./DictVector/dv-{date}.b')


#main(date="2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name='model_training',
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml']
)