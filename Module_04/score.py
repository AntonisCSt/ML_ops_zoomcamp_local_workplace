#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libriaries
import pickle
import sys
import pandas as pd
import numpy as np




# In[6]:

#needed categorical values
categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_csv(filename,sep='|')
    
    #convert to daytime
    df.dropOff_datetime = pd.to_datetime(df.dropOff_datetime)
    df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:

#Here we read the data import dv and lr(model) and perform the predicitons and create the df_result
def apply_model(input_file, run_id, output_file,year,month):

    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')

    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    #print mean duration
    print(f"Mean predicted duration: {np.mean(y_pred)}")
    
    df_result = pd.DataFrame()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['pickup_datetime'] = df['pickup_datetime']
    df_result['PULocationID'] = df['PUlocationID']
    df_result['DOLocationID'] = df['DOlocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    df_result.to_parquet(output_file,
    engine='pyarrow',
    compression=None,
    index=False)


# In[8]:
#pack run functions and needed comands
#example command (using pipenv): pipenv run python score.py fhv 2021 3 None 
def run():
    taxi_type = sys.argv[1] #'fhv'
    year = int(sys.argv[2]) #2021
    month = int(sys.argv[3]) #2
    run_id = sys.argv[4] #None
    
    input_file = f'../../data/{taxi_type}_tripdata_{year:04d}-{month:02d}.csv'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    apply_model(input_file, run_id, output_file,year,month)
    
if __name__ == '__main__':
    run()


# In[ ]:




