import numpy as np
import pandas as pd

def mid_cust_distance(mid, cid):
    """
    input merchant, customer: [lat, long], [lat,long] returns distance
    """
    mid = [x / 1e6 for x in mid]
    mid = np.radians(mid)
    cid = [x / 1e6 for x in cid]
    cid = np.radians(cid)

    a = np.sin((cid[0] - mid[0])/2)**2 + np.cos(mid[0]) * np.cos(cid[0]) * np.sin((cid[1]-mid[1])/2)**2 
    c = 2 * np.arcsin(np.sqrt(a)) 
    return 6371 * c

def get_features(df):
    """
    returns a set of features and target
    """
    featureset = df

    target = df["lateness"]

    featureset.drop(["dt","courier_id","dispatch_time","order_id","waybill_id","grab_time","fetch_time","arrive_time"],axis=1,inplace=True)

    featureset["distance"] = mid_cust_distance([featureset["sender_lat"],featureset["sender_lng"]],[featureset["recipient_lat"],featureset["recipient_lng"]])
    featureset.drop(["sender_lat","sender_lng","recipient_lat","recipient_lng","grab_lat","grab_lng"],axis=1,inplace=True)

    featureset["estimate_arrived_time"] = pd.to_datetime(featureset["estimate_arrived_time"], unit="s")
    featureset["estimate_meal_prepare_time"] = pd.to_datetime(featureset["estimate_meal_prepare_time"], unit="s")
    featureset["order_push_time"] = pd.to_datetime(featureset["order_push_time"], unit="s")
    featureset["platform_order_time"] = pd.to_datetime(featureset["platform_order_time"], unit="s")

    featureset["order_hour"] = (featureset["platform_order_time"].dt.hour)
    featureset["is_peak_hour"] = (featureset["platform_order_time"].dt.hour.isin([11,12,13,14,17,18,19,20]).astype(int))
    featureset["day_of_week"] = (featureset["platform_order_time"].dt.dayofweek)
    #featureset["is_weekend"] = (featureset["platform_order_time"].dt.dayofweek.isin([5,6]).astype(int))

    featureset["estimate_arrived_time"] = (featureset["estimate_arrived_time"] - featureset["platform_order_time"]).dt.total_seconds()/60
    featureset["estimate_meal_prepare_time"] = (featureset["estimate_meal_prepare_time"] - featureset["platform_order_time"]).dt.total_seconds()/60
    featureset["order_push_time"] = (featureset["order_push_time"] - featureset["platform_order_time"]).dt.total_seconds()/60
 
    featureset.drop("platform_order_time",inplace=True,axis=1)
    featureset.drop("lateness",inplace=True,axis=1)

    return featureset, target

