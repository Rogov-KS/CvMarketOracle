import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from help_src.tg_bot.logger import get_logger
from market_oracle_lib.data.t_bank import (
    find_instrument_by_ticker, get_symbol_data
)
from help_src.patterns.src import add_pivot_column, collect_channel, refine_channel, visualize_data
from help_src.patterns.src import PriceChannel
from dataclasses import dataclass
from help_src.gpt_api.perplexity_api import send_request as perp_send_request
from yandex_cloud_ml_sdk import YCloudML

from market_oracle_lib.model import LSTMModel
import torch
from market_oracle_lib.data.data_funcs import prepare_data, FEATURE_COLS
from market_oracle_lib.data import data_funcs
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from help_src.tg_bot.services.tsf_predict_service import make_prod_predict_by_cbt
import pandas as pd


logger = get_logger(__name__)

IMAGE_DIR = "logs/tmp_images"
CHECKPOINT_DIR = "checkpoints"


@dataclass
class PriceChannelAdvice:
    is_success: bool = False
    image_path: str | None = None
    text: str | None = None
    error: str | None = None


async def get_formatted_pattern_advice(ticker: str,
                                       t_token: str | None,
                                       granularity: str = "1d",
                                       window_len: int = 365,
                                       img_dir: str = IMAGE_DIR,
                                       do_refine: bool = False,
                                       add_oscilators: list[str] = [],
                                       ya_sdk: YCloudML | None = None) -> PriceChannelAdvice:
    """
    Получает исторические данные, строит ценовой канал,
    сохраняет график и возвращает путь к нему.
    """
    logger.info(f"get_formatted_pattern_advice: {ticker}, {t_token}, {granularity}, {window_len}, {img_dir}, {do_refine}, {add_oscilators}")
    if t_token is None:
        logger.error("Токен Tinkoff API не настроен для выполнения запроса.")
        return PriceChannelAdvice(error="Ошибка: Токен Tinkoff API не настроен.")

    os.makedirs(img_dir, exist_ok=True)
    logger.info(
        "Создана директория для изображений (если не существовала): %s",
        img_dir
    )

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_len)

        logger.info(
            "Запрос исторических данных для %s с %s по %s",
            ticker, start_date, end_date
        )

        try:
            data_df = get_symbol_data(
                symbol=ticker,
                token=t_token,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                interval=granularity,
            )
            data_count = len(data_df)
        except Exception as e_day:
            logger.warning(
                "Не удалось получить дневные данные для %s: %s. "
                "Пробуем часовые.",
                ticker, e_day
            )
            return PriceChannelAdvice(
                error=f"Ошибка: Не удалось получить данные для {ticker}."
            )

        if data_df is None or data_df.empty:
            logger.warning("Нет данных для %s за последний год.", ticker)
            return PriceChannelAdvice(
                error=f"Нет данных для {ticker} за последний год."
            )

        data_df = data_df.sort_values(by="Date", ascending=True).reset_index(
            drop=True
        )
        logger.info("Получено %s строк данных для %s", len(data_df), ticker)

        window_val = 3
        data_df = add_pivot_column(data_df, window_val)

        candle = len(data_df)
        backcandles = data_count // 3
        window = 3

        price_channel = collect_channel(
            df=data_df,
            candle=candle,
            backcandles=backcandles,
            window=window
        )

        if do_refine:
            price_channel = refine_channel(price_channel)

        logger.info(
            "Найден ценовой канал для %s",
            ticker
        )

        if not price_channel:
            logger.warning("Ценовые каналы не найдены для %s", ticker)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{ticker}_price_channel_{timestamp}.png"
        image_path = os.path.join(img_dir, image_filename)

        try:
            visualize_data(
                df=data_df,
                price_ch=price_channel,
                candle=candle,
                backcandles=backcandles,
                window=window,
                do_show=False,
                save_path=image_path,
                title=f"Ценовой канал для {ticker}",
                add_oscilators=add_oscilators
            )
        except Exception as e_img:
            logger.error(
                "Ошибка сохранения изображения %s: %s. Установите kaleido.",
                image_path, e_img
            )
            return PriceChannelAdvice(
                error=f"Ошибка сохранения графика для {ticker}. Установите kaleido."
            )

        logger.info("График сохранен: %s", image_path)
        try:
            pattern_advice_text = await get_formatted_pattern_advice_text(
                data_df=data_df,
                ticker=ticker,
                ya_sdk=ya_sdk
            )
            logger.info("pattern_advice_text: %s", pattern_advice_text)
        except Exception as e:
            logger.error("Ошибка при получении текста совета: %s", e)
            return PriceChannelAdvice(
                error=f"Ошибка при получении текста совета: {e}"
            )

        logger.info("pattern_advice_text: %s", pattern_advice_text)
        return PriceChannelAdvice(
            is_success=True,
            image_path=image_path,
            text=f"Анализ графика:\n{pattern_advice_text}",
            error=None
        )

    except Exception as e:
        logger.error(
            "Общая ошибка при построении ценового канала для %s: %s", ticker, e
        )
        return PriceChannelAdvice(
            error=f"Произошла ошибка при обработке запроса для {ticker}: {e}"
        )


SYSTEM_PROMPT_FOR_PATTERN_ADVICE = """
Ты умный ассистент, которому нужно проанализировать временной ряд цены акций компании и дать советы по покупке или продаже.
Не нужно упоминать об возможной не точности метода, люди и так это понимают.
Не надо писать 'Важно помнить, что анализ временных рядов не является гарантией будущих тенденций, и решения о покупке или продаже акций должны приниматься с учётом дополнительных факторов и рисков.'
Как раз, потому что люди понимают эти риски, так как анализ предоставляется осмысленным людям.
Временной ряд представлен в виде таблицы с колонками:
- Date - дата
- Open - цена открытия
- High - максимальная цена
- Low - минимальная цена
- Close - цена закрытия
- Volume - объем торгов
- Symbol - символ компании
- Pivot - цена пивот
- Points - является ли точка локальным максимумом или минимумом
"""

def make_query_text_from_df(data_df: pd.DataFrame, ticker: str) -> str:
    """
    Создает текст для запроса к LLM.
    """

    return f"Временной ряд компании {ticker}:\n{data_df.to_csv(sep='|', index=False)}"

async def get_formatted_pattern_advice_text(data_df: pd.DataFrame, ticker: str, ya_sdk: YCloudML) -> str:
    """
    Получает текст совета для временого ряда представленного текстом.
    """
    logger.info("FROM get_formatted_pattern_advice_text: %s", data_df.head(5))

    messages = [
        {
            "role": "system",
            "text": SYSTEM_PROMPT_FOR_PATTERN_ADVICE,
        },
        {
            "role": "user",
            "text": make_query_text_from_df(data_df, ticker),
        },
    ]
    logger.info("messages: %s", messages)
    model = ya_sdk.models.completions("yandexgpt").configure(temperature=0)
    result = model.run_deferred(messages).wait()
    logger.info("result: %s", result)
    return result.alternatives[0].text

def get_gpt_search_request(company_name: str) -> str:
    return f"""Найди последние новости по компании `{company_name}`
Какие- то их важные финансовые решения, а также информация про сектор их работы в целом

Дай краткий семантический анализ положения компании

Ответ дай в HTML-формате, а не в обычном Markdown-формате. Это очень важно
Используй только поддерживаемые теги: <b>, <i>, <u>, <s>, <code>, <pre>, <a>
Не используй теги <div>, <h>, <ul>, <li>"""

SYSTEM_PROMPT_FOR_IMPROVING_SEMANTIC_ADVICE = """Ты умный ассистент, которому нужно поправить ответ, который дал LLM.
Ответ является текстом с html-тегами, который потом будет отправлен в телеграм.
Ты должен исправить все ошибки в тексте, которые могут быть связаны с html-тегами.
Допустимыми html-тегами являются:
<b> - жирный текст
<i> - курсив
<s> - зачеркнутый текст
<u> - подчеркнутый текст
<a> - ссылка
<code> - Моноширинный текст
<pre> - мультистрочный моноширинный
<blockquote> - блок цитаты

Все остальные теги являются недопустимыми и должны быть удалены.
Особенно обрати внимание на теги: <ul>, <li>, <ol>, <p>, <h>, <h1>, <h2>, <h3>, <h4>, <h5>, <h6> - это всё запрещённые теги - их нужно удалить.
Единственной что теги <li> внутри теги <ul> можешь заменить на *, а внутри тега <ol> можешь заменить на порядковые цифры, начиная с 1.

Также если ты встречаешь markdown-разметку, например **, то её стоит заменить на html-теги <b> и </b>.

Также особое внимание обрати на незакрытые теги, точнее на любой текст, который может восприниматься как незакрытый тег - такой текст ты должен удалить.
Например если есть просто текст <pre> как открытый тег, но нет закрывающего тега </pre>, то такой текст ты должен удалить, и в выходном тексте не ддолжно быть тега <pre> вообще.
В ответ просто пришли исправленный текст, никаких комментариев и пояснений.
"""


async def improve_semantic_advice(advice: str, ya_sdk: YCloudML) -> str:
    """
    Исправляет семантический анализ компании.
    """
    messages = [
        {
            "role": "system",
            "text": SYSTEM_PROMPT_FOR_IMPROVING_SEMANTIC_ADVICE,
        },
        {
            "role": "user",
            "text": f"Ответ LLM, который нужно исправить:\n{advice}",
        },
    ]
    model = ya_sdk.models.completions("yandexgpt").configure(temperature=0)
    result = model.run_deferred(messages).wait()
    text_result = result.alternatives[0].text
    if text_result is not None:
        if text_result.startswith("```html"):
            text_result = text_result[len("```html"):]
        if text_result.endswith("```"):
            text_result = text_result[:-len("```")]

    logger.info("text_result: %s", text_result)

    return text_result


@dataclass
class SemanticAdvice:
    is_success: bool = False
    error: str | None = None
    text: str = ""


async def get_formatted_semantic_advice_with_gpt(ticker: str, t_token: str | None, ya_sdk: YCloudML) -> SemanticAdvice:
    """
    Получает совета по компании с помощью GPT.
    """
    if t_token is None:
        logger.error("Токен Tinkoff API не настроен для выполнения запроса.")
        return SemanticAdvice(
            error="Ошибка: Токен Tinkoff API не настроен для выполнения запроса."
        )

    logger.info(
        "Получение совета для компании %s и t_token is not None", ticker
    )

    instrument_info = find_instrument_by_ticker(ticker, t_token)
    logger.info("instrument_info:\n%s", instrument_info)

    gpt_search_req = get_gpt_search_request(instrument_info["name"])
    logger.info("gpt_search_req:\n%s", gpt_search_req)

    gpt_search_resp = await perp_send_request(gpt_search_req)
    gpt_search_resp_text = gpt_search_resp["choices"][0]["message"]["content"]
    logger.info("perplexity response:\n%s", gpt_search_resp_text)
    # total_resp = f"<b>Ответ perplexity:</b>\n{gpt_search_resp_text}"
    total_resp = await improve_semantic_advice(gpt_search_resp_text, ya_sdk)

    advice = SemanticAdvice(
        is_success=True,
        text=total_resp
    )
    return advice


@dataclass
class TimeSeriesForecastAdvice:
    is_success: bool = False
    error: str | None = None
    forecast: int = 0
    text: str = ""


async def get_formatted_time_series_forecast_advice(ticker: str,
                                                    t_token: str | None,
                                                    tsf_model: LSTMModel | None,
                                                    models: dict[str, CatBoostRegressor] | None,
                                                    scalers: dict[str, MinMaxScaler | StandardScaler] | None) -> TimeSeriesForecastAdvice:
    """
    Получает прогноз цены на 1 год вперед.
    """
    if t_token is None:
        logger.error("Токен Tinkoff API не настроен для выполнения запроса.")
        return "Ошибка: Токен Tinkoff API не настроен для выполнения запроса."

    logger.info(
        "Получение совета для компании %s и t_token is not None", ticker
    )

    instrument_info = find_instrument_by_ticker(ticker, t_token)
    logger.info("instrument_info:\n%s", instrument_info)

    # # Получение исторических данных
    # data_df = get_symbol_data(
    #     symbol=ticker,
    #     token=t_token,
    #     start_date=datetime.now() - timedelta(days=365),
    #     end_date=datetime.now(),
    #     interval="1d",
    # )

    # if data_df is None or data_df.empty:
    #     logger.warning("Нет данных для %s за последний год.", ticker)
    #     return "Ошибка: Нет данных для прогноза."

    # # logger.info("data_df:\n%s", data_df)

    # features = prepare_data(data_df)
    # print(f"{features.shape=}")
    # features = features.iloc[[-1]]
    # print(f"{features.shape=}")
    # features = features[FEATURE_COLS]
    # print(f"{features.shape=}")

    # logger.info("features:\n%s", features)
    # logger.info("features.shape:\n%s", features.shape)

    # with torch.no_grad():
    #     input_X = torch.tensor(features.values, dtype=torch.float32)
    #     logger.info("input_X:\n%s", input_X)
    #     forecast = tsf_model(input_X)
    #     logger.info("forecast:\n%s", forecast)

    forecast = make_prod_predict_by_cbt(ticker=ticker,
                                        loaded_models=models,
                                        saved_scalers=scalers,
                                        t_token=t_token)

    return TimeSeriesForecastAdvice(
        is_success=True,
        forecast=forecast,
        text=f"Прогноз цены на 1 неделю вперед: {forecast:.1f} рублей"
    )
