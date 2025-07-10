from __future__ import annotations
from yandex_cloud_ml_sdk import YCloudML
from dotenv import load_dotenv
import os

load_dotenv()

YA_TOKEN = os.getenv("YA_GPT_TOKEN")
YA_FOLDER_ID = os.getenv("YA_FOLDER_ID")

sdk = YCloudML(
    folder_id=YA_FOLDER_ID,
    auth=YA_TOKEN,
)

model = sdk.models.completions("yandexgpt")

messages_template = [
    {
        "role": "system",
        "text": "Найди ошибки в тексте и исправь их",
    }
]

SYSTEM_MESSAGE = "Найди ошибки в тексте и исправь их"


async def send_request(message: str,
                       system_message: str = SYSTEM_MESSAGE,
                       temperature: float = 0.5,
                       max_tokens: int = 500) -> str:
    messages = [
        {"role": "system", "text": system_message},
        {"role": "user", "text": message},
    ]

    conf_model = model.configure(temperature=temperature)
    operation = conf_model.run_deferred(messages, max_tokens=max_tokens)
    result = operation.wait()

    return result
