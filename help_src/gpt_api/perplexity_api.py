import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PERP_TOKEN")

url = "https://api.perplexity.ai/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

SYSTEM_MESSAGE = """Проведи семантический анализ новостей про компанию и выдай краткое резюме.
Последней строчкой должен быть ответ, каким является овостной фон компании: негативным, нейтральным или позитивным"""


async def send_request(message: str,
                       system_message: str = SYSTEM_MESSAGE,
                       temperature: float = 0.5,
                       max_tokens: int = 500) -> str:
    data = {
        "model": "sonar",  # Модель с доступом в интернет
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ],
        "temperature": temperature,  # Контроль случайности ответа (0–1)
        "max_tokens": max_tokens     # Максимальная длина ответа
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            response_json = await response.json()
            return response_json
