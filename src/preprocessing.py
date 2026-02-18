import pandas as pd

def getdata(path):

    df = pd.read_csv(r'data\all_waybill_info_meituan_0322.csv', index_col=0)
    train_data = df[0:int(len(df)*0.80)]
    test_data = df[int(len(df)*0.80):]

    print(df.shape)

    df = df[df['is_courier_grabbed'] != 0]
    df = df[df['arrive_time'] != 0]
    df = df[df['estimate_arrived_time'] != 0]
    df = df[df['grab_time'] != 0]
    df = df[df['estimate_meal_prepare_time'] != 0]
    df = df[df['dispatch_time'] != 0]
    df = df[df['grab_lat'] != 0]
    df = df[df['grab_lng'] != 0]

    return df
