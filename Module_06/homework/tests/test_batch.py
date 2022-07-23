import batch

from datetime import datetime
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def test_output():
    year = 2021
    month = 2
    y_pred = batch.main(year,month)
    pred_type = str(type(y_pred))
    assert "<class 'numpy.ndarray'>" == pred_type

def test_read_data():
    data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)
    categorical = ['PUlocationID', 'DOlocationID']
    df = batch.prepare_data(df,categorical)

   
    assert len(df) == 2
    assert round(df['duration'].iloc[0])== 8
    assert round(df['duration'].iloc[1])== 8


    

