from __future__ import annotations

import os
import datetime
import time
import logging
import json
import argparse

from typing import List, Optional, Tuple

from azure.identity import DefaultAzureCredential

from azure.storage.blob import ContainerClient
from azure.storage.queue import QueueClient

global logger
logger: logging.Logger

# upload davky ulozenej v lokalnom adresari na spracovanie
def upload_batch(container: ContainerClient, batch_folder: str, queue: QueueClient,
                 local_path: str, batch: str,
                 start: float, diff: int, sim_mode: bool):


    # cakame, kym mozeme poslat
    t = time.time()
    while sim_mode and t - start < diff:
        #logger.info('t: ' + str(t))
        logger.info('-- sleeping, time:{0:.3f}'.format(t - start))
        time.sleep(1)
        t = time.time()

    # upload datovych suborov z davky do containera
    folder = os.path.join(local_path, batch)
    for file_name in os.listdir(folder):
        if not file_name.endswith('.csv'):
            continue
        file_local_path = os.path.join(folder, file_name)

        # vyhladame blob podla mena
        blob_name = os.path.join(batch_folder, batch, file_name)
        blob_client = container.get_blob_client(blob=blob_name)
        with open(file=file_local_path, mode="rb") as data:
            if blob_client.exists(): # ak existuje, zmazeme ho
                blob_client.delete_blob()
            blob_client.upload_blob(data) # upload do containera
            logger.debug('upload blob: ' + blob_name)

    # zapis metadat do fronty
    md_fname = os.path.join(local_path, batch, 'metadata.json')
    md_file = open(md_fname, 'r')
    msg = md_file.read()
    queue.send_message(msg)
    logger.debug('msg sent to queue')

    logger.info('-- sending,  time:{0:.3f}'.format(t - start))


if __name__ == '__main__':

    # lokalna cesta k suborom
    base_path = '/home/azureuser/localfiles/v1'

    # --- konfiguracia ---
    STORAGE_ACCOUNT_NAME = "lmvzs8572858490"
    CONTAINER_NAME = "lmvzsblobstore-container"

    # logovanie
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
    parser.add_argument("--batch_folder", type=str, help="batch folder in blob container", default='batch')
    parser.add_argument("--batch_input", type=str, help="local folder with batchs", default='batch_input')
    parser.add_argument("--sim_mode", type=str, help="simulation mode", default=False)
    args = parser.parse_args()

    storage_account = args.storage_account
    container = args.container
    batch_folder = args.batch_folder
    batch_input = args.batch_input
    sim_mode = args.sim_mode

    my_credential = DefaultAzureCredential()

    # vytvorime klienta pre frontu
    queue_name = 'lmvzs-batch'
    queue_service_url = f"https://{storage_account}.queue.core.windows.net"
    queue = QueueClient(queue_service_url, queue_name=queue_name, credential=my_credential)
    logger.info("Using queue: " + queue_name)

    # vytvorime klienta pre blob container
    blob_service_url = f"https://{storage_account}.blob.core.windows.net"
    container = ContainerClient(
            account_url=blob_service_url,
            container_name=container,
            credential=my_credential)

    # zistime vsetky davky, ktore sa maju uploadovat
    source_batches: List[str] = []   # 'da_29098422'
    local_path = os.path.join(base_path, batch_input)

    # ak neboli definovane v source_batches, tak posleme cely vstupny adresar
    if len(source_batches) == 0:
        source_batches = os.listdir(local_path)
        source_batches = [s for s in source_batches if os.path.isdir(os.path.join(local_path, s))]
        # source_batches = source_batches[:10]

    # zistime poradie davok a posun davky od startu v sekundach
    start_dt = datetime.datetime.strptime('2024-01-13 01:00:00', '%Y-%m-%d %H:%M:%S')
    records: List[Tuple[str, int]] = []    # dvojice davka a jej posun od startu v sec
    for source_batch in source_batches:
        md_file = os.path.join(local_path, source_batch, 'metadata.json')
        f = open(md_file)
        d = json.load(f)
        datetime_str = d.get('datum_stavb')
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        diff = (datetime_object - start_dt).total_seconds()
        records.append((source_batch, int(diff)))

    # preusporiadanie podla datetime
    records.sort(key=lambda item: item[1])

    start_s = records[0][1]
    logger.info(f'Start uploading, start_s: {start_s}')

    # upload davok
    start = time.time()
    for source_batch, diff in records:
        logger.info('Upload ' + str(source_batch) + ' ' + str(diff - start_s))
        upload_batch(container, batch_folder, queue,
                     local_path, source_batch,
                     start, diff - start_s, sim_mode)

    end = time.time()
    logger.info('Upload OK: ' + str(end - start))
