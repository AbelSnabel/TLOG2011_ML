import numpy as np

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

    featureset.drop(["dt","courier_id","dispatch_time","order_id","waybill_id","grab_time","fetch_time","arrive_time"],axis=1,inplace=True)

    featureset["distance"] = mid_cust_distance([featureset["sender_lat"],featureset["sender_lng"]],[featureset["recipient_lat"],featureset["recipient_lng"]])

    featureset.drop(["sender_lat","sender_lng","recipient_lat","recipient_lng","grab_lat","grab_lng"],axis=1,inplace=True)

    target = df["lateness"]

    return featureset, target