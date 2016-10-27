
class Config(object):
    store_data_path = "data/agg_by_store.pickle"
    weather_data_path = "data/raw_store_weather_aggregated.csv"
    output_path = "output/"
    save_dir = "save/"
    test_size = 7
    output_layer = 2
    hidden_layers = [1024, 512, output_layer]
    batch_size = 128
    nb_epoch = 30
    dropout = 0.5
    loss_func = "mape"
    activation = "relu"
    optimizer = "rmsprop"

    feature_sizes = {
        "store_id": 1142,
        "state": 54,
        "isholiday": 2,
        "p_total_revenue": 1,
        "p_total_volume": 1,
        "dayofweek": 7,
        "dayofmonth": 31,
        "mean_temp": 1,
        "total_precipitation": 1,
        "total_snow": 1
    }

    embedding_sizes = {
        "store_id": 50,
        "state": 10,
        "isholiday": 2,
        "p_total_revenue": 1,
        "p_total_volume": 1,
        "dayofweek": 2,
        "dayofmonth": 3,
        "mean_temp": 1,
        "total_precipitation": 1,
        "total_snow": 1
    }