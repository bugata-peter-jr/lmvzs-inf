from __future__ import annotations
from typing import List, Optional

import os
import logging
import time
import argparse
import json

from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from azure.storage.queue import QueueClient

global logger
logger: logging.Logger

# upload davky ulozenej v lokalnom adresari na spracovanie
def download_dir(container: ContainerClient, output_folder: str,
                 local_path: str, batch: str) -> None:

    # vypis suborov pre download
    prefix = os.path.join(output_folder, batch)
    logger.info('prefix: ' + prefix)
    blob_list = list(container.list_blobs(name_starts_with=prefix))

    # download vsetkych suborov do lokalneho adresara pre davku
    download_dir_path = os.path.join(local_path, batch)
    if len(blob_list) > 0 and not os.path.exists(download_dir_path):
        os.mkdir(download_dir_path)

    for blob in blob_list:
        local_file_name = blob.name
        logger.info("blob: " + blob.name)
        idx = local_file_name.rfind('/')
        if idx >= 0:
            local_file_name = local_file_name[idx+1:]
        download_file_path = os.path.join(local_path, batch, local_file_name)
        with open(file=download_file_path, mode="wb") as download_file:
            download_file.write(container.download_blob(blob.name).readall())

    # odmazanie downloadovanych suborov
    for blob in blob_list:
        container.delete_blob(blob.name)
        logger.info('delete output blob: ' + blob.name)

    return

if __name__ == '__main__':

    # lokalna cesta k suborom
    base_path = '/home/azureuser/localfiles/v1'

    # --- konfiguracia ---
    STORAGE_ACCOUNT_NAME = "lmvzs8572858490"
    CONTAINER_NAME = "lmvzsblobstore-container"

    # logging
    log_level = logging.INFO

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.ERROR)

    logging.getLogger("azure").setLevel(logging.ERROR)
    logging.getLogger("azure.core").setLevel(logging.ERROR)
    logging.getLogger("azure.identity").setLevel(logging.ERROR)
    logging.getLogger("azure.storage.blob").setLevel(logging.ERROR)
    logging.getLogger("azure.storage.queue").setLevel(logging.ERROR)

    log_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(encoding='utf-8', level=log_level, format=log_format)

    logger = logging.getLogger(name=__name__)

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_account", type=str, help="path to batch folder", default=STORAGE_ACCOUNT_NAME)
    parser.add_argument("--container", type=str, help="path to batch folder", default=CONTAINER_NAME)
    parser.add_argument("--output_folder", type=str, help="output folder in blob container", default='output')
    parser.add_argument("--batch_output", type=str, help="local folder for batch outputs", default='batch_output')
    args = parser.parse_args()

    storage_account = args.storage_account
    container = args.container
    output_folder = args.output_folder
    batch_output = args.batch_output

    my_credential = DefaultAzureCredential()

    # vytvorime klienta pre frontu
    queue_name = 'lmvzs-output'
    queue_service_url = f"https://{storage_account}.queue.core.windows.net"
    queue = QueueClient(queue_service_url, queue_name=queue_name, credential=my_credential)
    logger.info("Using queue: " + queue_name)

    # vytvorime klienta pre blob container
    blob_service_url = f"https://{storage_account}.blob.core.windows.net"
    container = ContainerClient(
            account_url=blob_service_url,
            container_name=container,
            credential=my_credential)

    start = time.time()

    # precitame vsetky spravy z fronty
    n_batchs = 0
    while True:
        msg = queue.receive_message(visibility_timeout=30)
        if msg is None:
            break

        # mame vybranu davku
        content = msg.content
        json_dict = json.loads(content)
        batch = json_dict.get('iddavka')
        # !!!TODO osetrit parsovacie chyby
        local_path = os.path.join(base_path, batch_output)
        download_dir(container, output_folder, local_path, batch)
        logger.info(f'Download {batch} OK')

        queue.delete_message(msg)
        n_batchs += 1

    end = time.time()
    logger.info(f'n_batchs: {n_batchs}, duration: {end - start} s')
