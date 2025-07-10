from market_oracle_lib.data import data_funcs
from market_oracle_lib.data import t_bank
from market_oracle_lib.model import LSTMModel
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from IPython.display import clear_output
import pickle
from help_src.tg_bot.logger import get_logger
from IPython.display import display


logger = get_logger(__name__)


def make_prod_predict_by_cbt(ticker: str,
                      loaded_models: dict[str, CatBoostRegressor],
                      saved_scalers: dict[str, MinMaxScaler | StandardScaler],
                      t_token: str,
                      interval: str = "1d",
                      is_diff_target: bool = False):
    data_df, _ = data_funcs.create_featured_data_frame(t_bank.get_symbol_data,
                                                interval=interval,
                                                token=t_token,
                                                symbols_list=[ticker],
                                                t_token=t_token,
                                                scalers=saved_scalers,
                                                cols_for_scale=['Close', 'Open', 'High', 'Low'],
                                                do_drop_na=False)

    last_row = data_df.iloc[[-1]]
    logger.info(f"{last_row.columns[last_row.isna().sum(axis=0) > 0]=}, {last_row['Date']=} {last_row['Close']=}")
    input_row, _ = data_funcs.create_X_y_df_from_df(last_row, target_col='target')
    pred = loaded_models[ticker].predict(input_row)
    if is_diff_target:
        # display(last_row)
        pred = pred + last_row.iloc[0]['scaled_Close']
    scaled_pred = saved_scalers[ticker].inverse_transform(pred.reshape(-1, 1))[0, 0].item()
    return scaled_pred


def load_lstm_model(model_path: str,
                    input_size: int = 23,
                    hidden_size: int = 128,
                    output_size: int = 1) -> LSTMModel:
    model = LSTMModel.load_from_checkpoint(
        model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )
    model.eval()
    model.to('cpu')
    return model


def load_catboost_models(models_dir_path: str) -> dict[str, CatBoostRegressor]:
    loaded_models = {}
    for file_name in os.listdir(models_dir_path):
        ticker = file_name.split("_")[0]
        model_path = f"{models_dir_path}/{file_name}"
        if os.path.exists(model_path):
            model = CatBoostRegressor()
            model.load_model(model_path)
            loaded_models[ticker] = model
            print(f"Модель CatBoost для {ticker} загружена из {model_path}")
    # clear_output()
    return loaded_models


def load_scalers(scalers_dir_path: str) -> dict[str, MinMaxScaler | StandardScaler]:
    loaded_scalers = {}
    for file_name in os.listdir(scalers_dir_path):
        ticker = file_name.split("_")[0]
        scaler_path = f"{scalers_dir_path}/{file_name}"
        if os.path.exists(scaler_path):
            loaded_scalers[ticker] = pickle.load(open(scaler_path, 'rb'))
            print(f"{ticker} : mean = {loaded_scalers[ticker].mean_[0]:.1f} | scale = {loaded_scalers[ticker].scale_[0]:.1f} | path = {scaler_path}")

    return loaded_scalers
