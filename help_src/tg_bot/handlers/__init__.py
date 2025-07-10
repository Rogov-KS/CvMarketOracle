from aiogram import Router

from .common_handlers import router as common_router
from .asset_handlers import router as asset_router
from .advice_handlers import router as advice_router


main_router = Router()

main_router.include_router(asset_router)
main_router.include_router(advice_router)
main_router.include_router(common_router)
