#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import pandas as pd
import boto3

s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

options = {
    'client_kwargs': {
        'endpoint_url': s3_endpoint_url
                     }
              }
def create_s3_client():
    """
    This check if there is an s3 entry point defined. If not it will set a default one.
    """
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://s3:4566/')
    if s3_endpoint_url is None:
        return boto3.client('s3')    
    return boto3.client('s3')

def get_input_path(year:int, month:int):
    """
    This function gets the integers year and month and produces the input path (locally from data)
    """
    default_input_pattern = f'../../data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year:int, month:int):
    """
    This function gets the integers year and month and produces the out path s3 bucket path
    """
    default_output_pattern = 's3://nyc-duration/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def read_data(filename):
    """
    This function reads parquet data from the given file name and produces a dataframe
    """
    #s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    #options = {
    #'client_kwargs': {
    #    'endpoint_url': s3_endpoint_url
    #                 }
    #          }
    #print(filename)   
    #filename = 's3://nyc-duration/'+filename
    #df = pd.read_parquet(filename,storage_options=options)
    df = pd.read_parquet(filename)
    return df

def read_data_with_endpoint(year:int,month:int):
    """
    This function reads parquet data from the given file name in s3 bucket and produces a dataframe
    """
    filename = 'year={year:04d}month={month:02d}.parquet'
    filename = 's3://nyc-duration/'+filename
    df = pd.read_parquet(filename,storage_options=options)
    return df

def prepare_data(df:pd.DataFrame,categorical:list):
    """
    This fucntion applies processing steps to the dataframe we read.
    """
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def main(year,month):
    """
    This is the main function that uses the previous ones. 
    It will predict the taxi durarion and print the result
    """
    #input_file = f'../../data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'./results/fhv_tripdata_year={year:04d}-month={month:02d}-predictions.parquet'
    #input_file = get_input_path(year, month)
    #output_file = get_output_path(year, month)
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    categorical = ['PUlocationID', 'DOlocationID']
    #df = read_data(input_file)
    df = read_data_with_endpoint(year,month)
    df = prepare_data(df,categorical)
    #print(df)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    #df_result.to_parquet(output_file, engine='pyarrow', index=False)

    return y_pred

if __name__ == '__main__':
    year = 2021
    month = 1
    pred = main(year,month)
    #print(str(type(pred)))
    print(pred)