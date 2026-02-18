def getfeatures(df):
    featureset = df

    featureset.drop(["dt","courier_id","dispatch_time","order_id","waybill_id","grab_time","fetch_time","arrive_time"],axis=1,inplace=True)

    return featureset