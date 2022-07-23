import batch
from datetime import datetime
import pandas as pd
import os 

s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

options = {
    'client_kwargs': {
        'endpoint_url': s3_endpoint_url
                     }
              }

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def test_integration():
    data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    year = 2021
    month = 1
    input_file = 's3://nyc-duration/year={year:04d}month={month:02d}.parquet'
    df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
    )   
    
    #categorical = ['PUlocationID', 'DOlocationID']
    #df = batch.prepare_data(df,categorical)