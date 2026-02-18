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

    tolerance = 10    
    late_delivery = []
    for index, row in df.iterrows():
        row_lateness = [1 if row["arrive_time"] > (row["estimate_arrived_time"] + tolerance) else 0]
        late_delivery.append(row_lateness)
    late_delivery = pd.DataFrame(late_delivery)
    df = pd.concat([df,late_delivery],axis=1)


    train_data = df[0:int(len(df)*0.80)]
    test_data = df[int(len(df)*0.80):]

    return train_data, test_data, df


get_preprocess()