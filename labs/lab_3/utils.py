import pandas as pd

def get_trial_config(df:pd.DataFrame, trial_id:int):
    columns = df.columns
    param_cols = [col for col in columns if col.startswith("params")]
    trial_config = {}
    for col in param_cols:
        trial_config[col] = df[col][trial_id]
    
    return trial_config