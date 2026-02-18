import pandas as pd

def get_preprocess():
    """
    removes zero-values and splits data into train_data,test_data
    """
    path = r'data\all_waybill_info_meituan_0322.csv'
    df = pd.read_csv(path, index_col=0)


    df = df[df['is_courier_grabbed'] != 0]
    df = df[df['arrive_time'] != 0]
    df = df[df['estimate_arrived_time'] != 0]
    df = df[df['grab_time'] != 0]
    df = df[df['estimate_meal_prepare_time'] != 0]
    df = df[df['dispatch_time'] != 0]
    df = df[df['grab_lat'] != 0]
    df = df[df['grab_lng'] != 0]

    df = df.drop('is_courier_grabbed',axis=1)

    tolerance = 5*60

    df["lateness"] = (df["arrive_time"] > df["estimate_arrived_time"] + tolerance).astype(int)

    return df