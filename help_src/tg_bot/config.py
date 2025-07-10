import os
import dotenv
from dataclasses import dataclass


dotenv.load_dotenv()

# Замените "YOUR_TOKEN" на ваш токен
TOKEN = os.getenv("TG_TOKEN")
TINKOFF_TOKEN = os.getenv("T_TOKEN")

# Yandex Cloud ML
YA_FOLDER_ID = os.getenv("YA_FOLDER_ID")
YA_TOKEN = os.getenv("YA_GPT_TOKEN")


@dataclass
class Config:
    tg_token: str | None
    t_token: str | None
    ya_folder_id: str | None
    ya_token: str | None


# Пример создания конфига
config = Config(tg_token=TOKEN, t_token=TINKOFF_TOKEN,
                ya_folder_id=YA_FOLDER_ID, ya_token=YA_TOKEN)
