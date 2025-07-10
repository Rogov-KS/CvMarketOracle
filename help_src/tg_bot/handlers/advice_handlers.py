from aiogram import Router, types, F, Bot
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardMarkup
from aiogram.filters.callback_data import CallbackData
from yandex_cloud_ml_sdk import YCloudML
from help_src.tg_bot.services.advice_service import (
    IMAGE_DIR,
    get_formatted_pattern_advice,
    get_formatted_semantic_advice_with_gpt,
    get_formatted_time_series_forecast_advice
)
from help_src.tg_bot.config import Config
from help_src.tg_bot.logger import get_logger
from market_oracle_lib.model import LSTMModel
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


logger = get_logger(__name__)

router = Router()


class AdviceCallbackData(CallbackData, prefix="advice"):
    """Фабрика CallbackData для советов"""
    action: str  # "give_advice" или "other_company"
    ticker: str | None = None  # Тикер компании или None


class AdviceStates(StatesGroup):
    """Состояния для отслеживания ввода пользователя"""
    waiting_for_company_name = State()
    waiting_for_period = State()
    waiting_for_granularity = State()
    waiting_for_indicators = State()  # Изменено название состояния


def _create_company_selection_keyboard(action_name: str = "give_advice") -> InlineKeyboardMarkup:
    """Создает клавиатуру для выбора компании."""
    builder = InlineKeyboardBuilder()
    companies = {
        "Сбер Банк (SBER)": "SBER",
        "ЛУКОЙЛ (LKOH)": "LKOH",
        "Роснефть (ROSN)": "ROSN",
        "НОВАТЭК (NVTK)": "NVTK",
        "Газпром (GAZP)": "GAZP",
        "Газпром нефть (SIBN)": "SIBN",
        "Полюс (PLZL)": "PLZL",
        "Норильский никель (GMKN)": "GMKN",
        "Яндекс (YDEX)": "YDEX",
        "Татнефть (TATN)": "TATN",
        "Т-Технологии (T)": "T",
        "ФосАгро (PHOR)": "PHOR",
        # "Северсталь (CHMF)": "CHMF",
    }
    #     "Сургутнефтегаз (SNGS)": "SNGS",
    #     "ФосАгро (PHOR)": "PHOR",
    #     "НЛМК (NLMK)": "NLMK",
    #     "Аэрофлот (AFLT)": "AFLT",
    #     "ПИК (PIKK)": "PIKK",
    # }
    for name, ticker in companies.items():
        builder.button(
            text=name,  # Button text will be "Company Name (TICKER)"
            callback_data=AdviceCallbackData(
                action=action_name, ticker=ticker
            ).pack()
        )
    builder.adjust(3)  # Arrange buttons in 3 columns

    # builder.row(  # Кнопка "Другая компания" в отдельном ряду
    #     types.InlineKeyboardButton(
    #         text="Другая компания",
    #         callback_data=AdviceCallbackData(action="other_company").pack()
    #     )
    # )
    return builder.as_markup()


def _create_granularity_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру для выбора гранулярности."""
    builder = InlineKeyboardBuilder()
    builder.button(text="Дни", callback_data="granularity_days")
    builder.button(text="Часы", callback_data="granularity_hours")
    builder.adjust(2)
    return builder.as_markup()


def _create_indicator_keyboard(selected_indicators: list[str] = None) -> InlineKeyboardMarkup:
    """Создает клавиатуру для выбора индикаторов."""
    if selected_indicators is None:
        selected_indicators = []

    builder = InlineKeyboardBuilder()
    indicators = ["MACD", "SMA", "EMA", "RSI"]

    # Добавляем кнопки индикаторов
    for indicator in indicators:
        # Добавляем галочку к выбранным индикаторам
        text = f"✓ {indicator}" if indicator in selected_indicators else indicator
        builder.button(
            text=text,
            callback_data=f"toggle_indicator_{indicator}"
        )

    # Добавляем кнопку "Готово" только если есть выбранные индикаторы
    if selected_indicators:
        builder.button(
            text="✅ Готово",
            callback_data="finish_indicators"
        )

    builder.adjust(2)  # Располагаем кнопки в 2 колонки
    return builder.as_markup()


@router.message(Command(commands=["give_advice"]))
async def give_advice_command(message: types.Message, state: FSMContext):
    """Обработчик команды /give_advice."""
    keyboard = _create_company_selection_keyboard(action_name="give_advice")
    await message.answer(
        "Выберите компанию для получения совета:",
        reply_markup=keyboard
    )


@router.callback_query(AdviceCallbackData.filter(F.action == "give_advice"))
async def process_company_callback(
    callback_query: types.CallbackQuery,
    callback_data: AdviceCallbackData,
    state: FSMContext
):
    """Обработчик нажатия на кнопку с названием компании."""
    if callback_data.ticker is None:
        await callback_query.answer("Ошибка: тикер компании не найден.")
        return

    # Сохраняем выбранный тикер в состоянии
    await state.update_data(ticker=callback_data.ticker)

    # Запрашиваем период
    await callback_query.message.answer(
        "Введите количество дней для анализа (целое число):"
    )
    await state.set_state(AdviceStates.waiting_for_period)
    await callback_query.answer()


@router.message(AdviceStates.waiting_for_period)
async def process_period_input(message: types.Message, state: FSMContext):
    """Обработчик ввода периода."""
    try:
        period = int(message.text)
        if period <= 0:
            raise ValueError("Период должен быть положительным числом")

        # Сохраняем период в состоянии
        await state.update_data(period=period)

        # Запрашиваем гранулярность
        keyboard = _create_granularity_keyboard()
        await message.answer(
            "Выберите гранулярность:",
            reply_markup=keyboard
        )
        await state.set_state(AdviceStates.waiting_for_granularity)
    except ValueError:
        await message.answer(
            "Пожалуйста, введите корректное целое положительное число."
        )


@router.callback_query(F.data.startswith("granularity_"))
async def process_granularity_callback(
    callback_query: types.CallbackQuery,
    state: FSMContext,
):
    """Обработчик выбора гранулярности."""
    await callback_query.answer()

    granularity = callback_query.data.split("_")[1]  # "days" или "hours"
    if granularity == "days":
        real_granularity = "1d"
    elif granularity == "hours":
        real_granularity = "1h"
    else:
        raise ValueError(f"Неизвестная гранулярность: {granularity}")
    logger.info(f"process_granularity_callback: {real_granularity=}")

    # Сохраняем гранулярность в состоянии
    await state.update_data(granularity=real_granularity)

    # Инициализируем пустой список выбранных индикаторов
    await state.update_data(selected_indicators=[])

    # Запрашиваем выбор индикаторов
    keyboard = _create_indicator_keyboard()
    await callback_query.message.answer(
        "Выберите индикаторы для анализа (можно выбрать несколько):",
        reply_markup=keyboard
    )
    await state.set_state(AdviceStates.waiting_for_indicators)


@router.callback_query(F.data.startswith("toggle_indicator_"))
async def process_indicator_toggle(
    callback_query: types.CallbackQuery,
    state: FSMContext,
):
    """Обработчик переключения индикатора."""
    await callback_query.answer()

    indicator = callback_query.data.split("_")[2]  # Получаем название индикатора

    # Получаем текущий список выбранных индикаторов
    data = await state.get_data()
    selected_indicators = data.get("selected_indicators", [])

    # Переключаем состояние индикатора
    if indicator in selected_indicators:
        selected_indicators.remove(indicator)
    else:
        selected_indicators.append(indicator)

    # Обновляем список в состоянии
    await state.update_data(selected_indicators=selected_indicators)

    # Обновляем клавиатуру
    keyboard = _create_indicator_keyboard(selected_indicators)
    await callback_query.message.edit_text(
        "Выберите индикаторы для анализа (можно выбрать несколько):",
        reply_markup=keyboard
    )


@router.callback_query(F.data == "finish_indicators")
async def process_indicators_finish(
    callback_query: types.CallbackQuery,
    state: FSMContext,
    config: Config,
    bot: Bot,
    ya_sdk: YCloudML,
    tsf_model: LSTMModel | None,
    models: dict[str, CatBoostRegressor] | None,
    scalers: dict[str, MinMaxScaler | StandardScaler] | None
):
    """Обработчик завершения выбора индикаторов."""
    await callback_query.answer()

    # Получаем все сохраненные данные
    data = await state.get_data()
    ticker = data.get("ticker")
    period = data.get("period")
    granularity = data.get("granularity")
    selected_indicators = data.get("selected_indicators", [])

    if not all([ticker, period, granularity]) or not selected_indicators:
        await callback_query.message.answer("Произошла ошибка. Попробуйте снова.")
        await state.clear()
        return

    logger.info(
        f"\nПолученные параметры:\n"
        f"ticker: {ticker}, period: {period}, "
        f"granularity: {granularity}, "
        f"indicators: {selected_indicators}\n"
    )

    # Отправляем сообщение о прогрессе
    progress_message = await bot.send_message(
        chat_id=callback_query.message.chat.id,
        text="Пожалуйста, подождите..."
    )

    # PATTERN ADVICE
    advice_text = await get_formatted_pattern_advice(
        ticker, config.t_token,
        window_len=period,
        granularity=granularity,
        add_oscilators=selected_indicators,
        img_dir=IMAGE_DIR,
        do_refine=True,
        ya_sdk=ya_sdk
    )

    if advice_text.is_success and advice_text.image_path:
        await progress_message.delete()
        await bot.send_photo(
            chat_id=callback_query.message.chat.id,
            photo=FSInputFile(advice_text.image_path)
        )

        await bot.send_message(
            chat_id=callback_query.message.chat.id,
            text=advice_text.text
        )
    else:
        await progress_message.edit_text(advice_text.error or "Произошла ошибка")

    # SEMANTIC ADVICE
    advice_text = await get_formatted_semantic_advice_with_gpt(
        ticker, config.t_token,
        ya_sdk=ya_sdk
    )

    if advice_text.is_success:
        logger.info(f"\nsemantic advice:\n{advice_text.text}\n")
        await bot.send_message(
            chat_id=callback_query.message.chat.id,
            text=advice_text.text,
            parse_mode="HTML"
        )
    else:
        logger.info(f"semantic advice: {advice_text.error}")
        await bot.send_message(
            chat_id=callback_query.message.chat.id,
            text=advice_text.error or "Произошла ошибка"
        )

    # TIME SERIES FORECAST ADVICE
    advice_text = await get_formatted_time_series_forecast_advice(
        ticker, config.t_token,
        tsf_model=tsf_model,
        models=models,
        scalers=scalers,
    )

    if advice_text.is_success:
        await bot.send_message(
            chat_id=callback_query.message.chat.id,
            text=advice_text.text,
            parse_mode="HTML"
        )
    else:
        await bot.send_message(
            chat_id=callback_query.message.chat.id,
            text=advice_text.error or "Произошла ошибка"
        )

    await state.clear()


# @router.message(AdviceStates.waiting_for_company_name)
# async def process_user_company_name(
#     message: types.Message, state: FSMContext, config: Config
# ):
#     """Обработчик ввода названия компании пользователем."""
#     company_name = message.text
#     if company_name is None:
#         await message.answer(
#             "Название компании не может быть пустым. Попробуйте еще раз."
#         )
#         return

#     response_text = await get_formatted_advice(company_name, config.t_token)
#     await message.answer(response_text)
#     await state.clear()


@router.message(Command(commands=["pattern_advice"]))
async def pattern_advice_command(message: types.Message):
    """Обработчик команды /pattern_advice."""
    keyboard = _create_company_selection_keyboard(action_name="pattern_advice")
    await message.answer(
        "Выберите компанию для получения паттерн-совета:",
        reply_markup=keyboard
    )


@router.message(Command(commands=["semantic_advice"]))
async def semantic_advice_command(message: types.Message):
    """Обработчик команды /semantic_advice."""
    keyboard = _create_company_selection_keyboard(action_name="semantic_advice")
    await message.answer(
        "Выберите компанию для получения семантического совета:",
        reply_markup=keyboard
    )


@router.message(Command(commands=["forecast_advice"]))
async def forecast_advice_command(message: types.Message):
    """Обработчик команды /forecast_advice."""
    keyboard = _create_company_selection_keyboard(action_name="forecast_advice")
    await message.answer(
        "Выберите компанию для получения прогнозного совета:",
        reply_markup=keyboard
    )


@router.callback_query(AdviceCallbackData.filter(F.action == "pattern_advice"))
async def process_pattern_advice_callback(
    callback_query: types.CallbackQuery,
    callback_data: AdviceCallbackData,
    config: Config,
    bot: Bot
):
    """Обработчик нажатия на кнопку с названием компании для паттерн-совета."""
    chat_id = callback_query.message.chat.id if callback_query.message else None
    if chat_id is None:
        await callback_query.answer()
        return

    progress_message = await bot.send_message(
        chat_id=chat_id,
        text="Пожалуйста, подождите..."
    )

    if callback_data.ticker is None:
        await progress_message.edit_text(
            "Ошибка: тикер компании не найден."
        )
        await callback_query.answer()
        return

    advice_text = await get_formatted_pattern_advice(
        callback_data.ticker, config.t_token
    )

    if advice_text.is_success and advice_text.image_path:
        await progress_message.delete()
        await bot.send_photo(
            chat_id=chat_id,
            photo=FSInputFile(advice_text.image_path)
        )
    else:
        await progress_message.edit_text(advice_text.error)

    await callback_query.answer()


@router.callback_query(AdviceCallbackData.filter(F.action == "semantic_advice"))
async def process_semantic_advice_callback(
    callback_query: types.CallbackQuery,
    callback_data: AdviceCallbackData,
    config: Config,
    bot: Bot
):
    """Обработчик нажатия на кнопку с названием компании для семантического совета."""
    chat_id = callback_query.message.chat.id if callback_query.message else None
    if chat_id is None:
        await callback_query.answer()
        return

    progress_message = await bot.send_message(
        chat_id=chat_id,
        text="Пожалуйста, подождите..."
    )

    if callback_data.ticker is None:
        await progress_message.edit_text(
            "Ошибка: тикер компании не найден."
        )
        await callback_query.answer()
        return

    advice_text = await get_formatted_semantic_advice_with_gpt(
        callback_data.ticker, config.t_token
    )

    if advice_text.is_success:
        await progress_message.edit_text(advice_text.text, parse_mode="HTML")
    else:
        await progress_message.edit_text(advice_text.error)

    await callback_query.answer()


@router.callback_query(AdviceCallbackData.filter(F.action == "forecast_advice"))
async def process_forecast_advice_callback(
    callback_query: types.CallbackQuery,
    callback_data: AdviceCallbackData,
    config: Config,
    bot: Bot,
    tsf_model: LSTMModel | None,
    models: dict[str, CatBoostRegressor] | None,
    scalers: dict[str, MinMaxScaler | StandardScaler] | None
):
    """Обработчик нажатия на кнопку с названием компании для прогнозного совета."""
    chat_id = callback_query.message.chat.id if callback_query.message else None
    if chat_id is None:
        await callback_query.answer()
        return

    progress_message = await bot.send_message(
        chat_id=chat_id,
        text="Пожалуйста, подождите..."
    )

    if callback_data.ticker is None:
        await progress_message.edit_text(
            "Ошибка: тикер компании не найден."
        )
        await callback_query.answer()
        return

    advice_text = await get_formatted_time_series_forecast_advice(
        callback_data.ticker, config.t_token,
        tsf_model=tsf_model,
        models=models,
        scalers=scalers
    )

    if advice_text.is_success:
        await progress_message.edit_text(advice_text.text, parse_mode="HTML")
    else:
        await progress_message.edit_text(advice_text.error)

    await callback_query.answer()
