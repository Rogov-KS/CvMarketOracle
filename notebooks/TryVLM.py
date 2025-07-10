#!/usr/bin/env python3

from __future__ import annotations
import pathlib
from yandex_cloud_ml_sdk import YCloudML

PATH = pathlib.Path(__file__)
NAME = f'example-{PATH.parent.name}-{PATH.name}'


def local_path(path: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent / path


def main() -> None:
    sdk = YCloudML(
        folder_id="b1g9sfneid72b27satue",
        auth="AQVNwdEXsQuvuSb17EaKsQ2Qr4nsx08S6ddRxHdC",
    )

    sdk.setup_default_logging()

    # model = sdk.models.completions('qwen2.5-vl-7b-instruct')

    # # Пакетный запуск вернет объект Operations
    # # Вы можете отслеживать его статус или вызвать метод .wait
    # operation = model.batch.run_deferred("fdsecird61j3i2vaigln")

    # resulting_dataset = operation.wait()

    # # Датасет с результатами вернется в формате Parquet
    # try:
    #     import pyarrow

    #     print('Resulting dataset lines:')
    #     for line in resulting_dataset.read():
    #         print(line)
    # except ImportError:
        # print('skipping dataset read; install yandex-cloud-ml-sdk[datasets] to be able to read')

if __name__ == '__main__':
    main()
