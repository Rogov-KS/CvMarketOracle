import asyncio
from tinkoff.invest import Client
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest.schemas import InstrumentStatus
from market_oracle_lib.data.t_bank import _get_instrument_methods
from help_src.utils.sort_by_capital import sort_shares_by_market_cap
from help_src.tg_bot.logger import get_logger

logger = get_logger(__name__)

MAX_INSTRUMENTS_TO_SHOW_PER_CATEGORY = 5
MAX_RUSSIAN_SHARES_TO_SHOW = 20


def _fetch_and_format_assets_sync(token: str) -> str:
    """
    Синхронная функция для получения и форматирования списка активов.
    Вызывается в отдельном потоке.
    """
    output_lines = ["<b>Сводный список доступных активов:</b>\n"]
    processed_categories = 0

    try:
        with Client(token, target=INVEST_GRPC_API) as services:
            instrument_method_tuples = _get_instrument_methods(services)

            for category_name, method_to_call in instrument_method_tuples:
                output_lines.append(
                    f"\n<b>--- {category_name.capitalize()} ---</b>"
                )
                processed_categories += 1
                try:
                    response = method_to_call(
                        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
                    )
                    instruments_in_category = []
                    if hasattr(response, 'instruments') and response.instruments:
                        # Получение списка инструментов в категории
                        instruments_in_category = response.instruments

                    if instruments_in_category:
                        for i, instrument_data in enumerate(
                            instruments_in_category[:MAX_INSTRUMENTS_TO_SHOW_PER_CATEGORY]
                        ):
                            output_lines.append(
                                f"- {instrument_data.name} "
                                f"({instrument_data.ticker})"
                            )
                        remaining_count = (
                            len(instruments_in_category) -
                            MAX_INSTRUMENTS_TO_SHOW_PER_CATEGORY
                        )
                        if remaining_count > 0:
                            output_lines.append(
                                f"  <em>...и еще {remaining_count}</em>"
                            )
                    else:
                        output_lines.append(
                            "  <em>(нет торгуемых инструментов "
                            "в этой категории)</em>"
                        )
                except Exception as e:
                    err_msg = (
                        f"Ошибка категории '{category_name}': "
                        f"{type(e).__name__} - {str(e)[:50]}"
                    )
                    logger.error(err_msg)
                    output_lines.append(
                        f"  <em>(ошибка загрузки: {type(e).__name__})</em>"
                    )
    except Exception as e:
        general_err_msg = (
            f"Общая ошибка в _fetch_and_format_assets_sync: "
            f"{type(e).__name__} - {str(e)[:100]}"
        )
        logger.error(general_err_msg)
        if processed_categories == 0:
            return (
                "Не удалось подключиться к сервису для получения "
                "списка активов. Пожалуйста, попробуйте позже."
            )
        output_lines.append(
            "\n<em>Произошла ошибка при загрузке полного списка активов. "
            "Показана частичная информация.</em>"
        )

    if processed_categories == 0 and len(output_lines) <= 1:
        return (
            "Не удалось загрузить информацию ни по одной категории активов. "
            "Возможно, сервис недоступен."
        )

    return "\n".join(output_lines)


def _fetch_and_format_russian_shares_sync(token: str) -> str:
    """
    Синхронная функция для получения и форматирования списка
    исключительно российских акций.
    Вызывается в отдельном потоке.
    """
    output_lines = ["<b>Список российских акций:</b>\n"]
    processed_any_shares = False

    try:
        with Client(token, target=INVEST_GRPC_API) as services:
            try:
                response = services.instruments.shares(
                    instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
                )
                russian_shares_full_list = []
                if hasattr(response, 'instruments') and response.instruments:
                    russian_shares_full_list = [
                        share for share in response.instruments
                        if share.country_of_risk == "RU"
                    ]

                # Добавляем сортировку акций по рыночной капитализации
                if russian_shares_full_list:
                    try:
                        russian_shares_full_list = sort_shares_by_market_cap(
                            russian_shares_full_list, token
                        )
                    except Exception as sort_exc:
                        logger.error(
                            f"Ошибка при сортировке российских акций: "
                            f"{type(sort_exc).__name__} - {str(sort_exc)[:100]}"
                        )
                        # Продолжаем с несортированным списком, если сортировка не удалась

                if russian_shares_full_list:
                    processed_any_shares = True
                    shares_to_display = (
                        russian_shares_full_list[:MAX_RUSSIAN_SHARES_TO_SHOW]
                    )
                    for share_data in shares_to_display:
                        output_lines.append(
                            f"- {share_data.name} ({share_data.ticker})"
                        )
                    if len(russian_shares_full_list) > MAX_RUSSIAN_SHARES_TO_SHOW:
                        remaining = (
                            len(russian_shares_full_list) -
                            MAX_RUSSIAN_SHARES_TO_SHOW
                        )
                        output_lines.append(
                            f"  <em>...и еще {remaining} "
                            f"российских акций</em>"
                        )
                else:
                    output_lines.append(
                        "  <em>(Нет доступных российских акций)</em>"
                    )
                    processed_any_shares = True
            except Exception as e:
                err_msg = (
                    f"Ошибка при загрузке российских акций: "
                    f"{type(e).__name__} - {str(e)[:50]}"
                )
                logger.error(err_msg)
                output_lines.append(
                    f"  <em>(Ошибка при загрузке: {type(e).__name__})</em>"
                )
    except Exception as e:
        general_err_msg = (
            f"Общая ошибка в _fetch_and_format_russian_shares_sync: "
            f"{type(e).__name__} - {str(e)[:100]}"
        )
        logger.error(general_err_msg)
        if not processed_any_shares:
            return (
                "Не удалось подключиться к сервису для получения списка "
                "российских акций. Пожалуйста, попробуйте позже."
            )
        output_lines.append(
            "\n<em>Произошла ошибка при загрузке списка российских акций.</em>"
        )

    if not processed_any_shares and len(output_lines) <= 1:
        return (
            "Не удалось загрузить информацию о российских акциях. "
            "Возможно, сервис недоступен."
        )

    return "\n".join(output_lines)


async def get_formatted_assets_list(token: str) -> str:
    """
    Асинхронная обертка для вызова _fetch_and_format_assets_sync.
    """
    # return await asyncio.to_thread(_fetch_and_format_assets_sync, token)
    return await asyncio.to_thread(_fetch_and_format_russian_shares_sync,
                                   token)
