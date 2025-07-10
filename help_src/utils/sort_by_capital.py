from typing import List, Dict

from tinkoff.invest import Client, Share, GetAssetFundamentalsRequest
from tinkoff.invest.constants import INVEST_GRPC_API


FUNDAMENTALS_BATCH_SIZE = 100  # Максимум для запроса GetAssetFundamentals


def _fetch_market_caps_sync(token: str, asset_uids: List[str]) -> Dict[str, float]:
    """
    Синхронная функция для получения рыночной капитализации.
    Вызывается в отдельном потоке.
    """
    market_caps: Dict[str, float] = {}
    if not asset_uids:
        return market_caps

    try:
        with Client(token, target=INVEST_GRPC_API) as services:
            for i in range(0, len(asset_uids), FUNDAMENTALS_BATCH_SIZE):
                batch_uids = asset_uids[i:i + FUNDAMENTALS_BATCH_SIZE]
                if not batch_uids:
                    continue
                try:
                    req = GetAssetFundamentalsRequest(assets=batch_uids)
                    fund_resp = services.instruments.get_asset_fundamentals(
                        request=req
                    )
                    for fundamental_data in fund_resp.fundamentals:
                        # Проверяем наличие market_capitalization
                        if (
                            hasattr(fundamental_data, 'market_capitalization') and
                            fundamental_data.market_capitalization is not None
                        ):
                            market_caps[fundamental_data.asset_uid] = (
                                fundamental_data.market_capitalization
                            )
                except Exception as e:
                    print(
                        f"Ошибка получения фундам. данных для батча "
                        f"{i//FUNDAMENTALS_BATCH_SIZE}: {type(e).__name__} - {e}"
                    )
    except Exception as e:
        print(
            f"Общая ошибка в _fetch_market_caps_sync: "
            f"{type(e).__name__} - {str(e)[:100]}"
        )

    return market_caps


def sort_shares_by_market_cap(
    shares: List[Share], token: str
) -> List[Share]:
    """
    Получает капитализацию для списка акций и сортирует их по убыванию.

    Args:
        shares: Список объектов tinkoff.invest.Share.
        token: Токен Tinkoff API.

    Returns:
        Новый список Share, отсортированный по капитализации (убывание).
        Акции без данных о капитализации будут в конце.
    """
    asset_uids = [share.asset_uid for share in shares if share.asset_uid]
    if not asset_uids:
        return shares

    market_caps = _fetch_market_caps_sync(token, asset_uids)

    # Сортируем список акций, используя полученные капитализации
    sorted_shares = sorted(
        shares,
        # -1 для акций без данных о капитализации
        key=lambda share: market_caps.get(share.asset_uid, -1.0),
        reverse=True
    )

    return sorted_shares
