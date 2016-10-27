from models import run_separate_model
from ensemble_model import run_lr
import utils
from configuration import Config

def train_ffnn_models():
    # List of list of numeric vars to keep
    num_var2keeps = [
        ["mean_temp", "total_precipitation", "total_snow", "p_total_revenue",
         "p_total_volume"],
        ["mean_temp", "p_total_revenue", "p_total_volume"],
        ["total_precipitation", "p_total_revenue", "p_total_volume"],
        ["total_snow", "p_total_revenue", "p_total_volume"],
        ["mean_temp", "total_precipitation", "total_snow", "p_total_revenue"],
        ["mean_temp", "total_precipitation", "total_snow", "p_total_volume"],
    ]

    # List of list of categorical vars to keep
    cat_var2keeps = [
        ["store_id", "state", "isholiday", "dayofweek", "dayofmonth"],
        ["state", "isholiday", "dayofweek", "dayofmonth"],
        ["store_id", "isholiday", "dayofweek", "dayofmonth"],
        ["store_id", "state", "isholiday"],
        ["store_id", "state", "dayofweek"],
        ["store_id", "state", "dayofmonth"],
    ]

    # Train each FFNN separately
    model_idx = -1
    for embedding in [True, False]:
        for num_list in num_var2keeps:
            for cat_list in cat_var2keeps:
                model_idx += 1
                model = run_separate_model(model_idx=model_idx,
                                           embedding=embedding,
                                           num_var2keeps=num_list,
                                           cat_var2keeps=cat_list)

    return model_idx


def run_ensemble(model_idx):
    return run_lr(model_idx)


def main():
    # Run the main application
    model_idx = train_ffnn_models()
    mse, mae, mape = run_ensemble(model_idx + 1)

    results = {"mse": mse, "mae": mae, "mape": mape}
    print results

    utils.to_pickle(Config.output_path + "results", results)


if __name__=="__main__":
    main()