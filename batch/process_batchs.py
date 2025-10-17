import os
import sys
import time
import argparse
import logging
import datetime
import json
import tempfile

from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.storage.queue import QueueClient, QueueMessage
from azure.storage.blob import ContainerClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(SCRIPT_DIR) not in sys.path:
    sys.path.append(os.path.dirname(SCRIPT_DIR))
#print('sys.path.append:', os.path.dirname(SCRIPT_DIR))

#from scoring_script import Scoring
from autoencoder.scoring_outliers import Scoring

class BatchEndpoint(object):

    def __init__(self, model_path: str,     # cesta k modelom
                    storage_account: str,   # storage account
                    container: str,         # container (datastore)
                    batch_folder: str,      # folder v datastore pre davky
                    output_folder: str,     # folder v datastore pre vysledky spracovania
                    batch_queue: str,       # queue pre davky
                    output_queue: str):     # queue pre vysledky spracovania

        self.model_path = model_path
        self.batch_folder = batch_folder
        self.output_folder = output_folder
        self.batch_queue = batch_queue
        self.output_queue = output_queue

        # logging
        log_level = logging.INFO

        root = logging.getLogger()
        root.handlers.clear()

        logging.getLogger("azure").setLevel(logging.ERROR)
        logging.getLogger("azure.core").setLevel(logging.ERROR)
        logging.getLogger("azure.identity").setLevel(logging.ERROR)
        logging.getLogger("azure.storage.blob").setLevel(logging.ERROR)
        logging.getLogger("azure.storage.queue").setLevel(logging.ERROR)
        logging.getLogger("azureml").setLevel(logging.ERROR)

        log_format = '%(asctime)s %(levelname)s: %(message)s'
        logging.basicConfig(encoding='utf-8', level=log_level, format=log_format)

        self.logger = logging.getLogger(name=__name__)

        # credential
        self.credential = DefaultAzureCredential()

        # queues
        queue_service_url = f"https://{storage_account}.queue.core.windows.net"
        self.batch_queue = QueueClient(queue_service_url, queue_name=self.batch_queue, credential=self.credential)
        self.output_queue = QueueClient(queue_service_url, queue_name=self.output_queue, credential=self.credential)
        self.visibility_timeout = 30

        # blobs
        blob_service_url = f"https://{storage_account}.blob.core.windows.net"
        self.container = ContainerClient(
                account_url=blob_service_url,
                container_name=container,
                credential=self.credential)

        # init
        self.scoring = Scoring(self.logger)
        self.scoring.load_model(model_path)

    # caka na batch, kym nie je dostupny
    def wait_for_batch(self, pooling_time=10) -> QueueMessage:

        while True:
            # kontrola na vyskyt davok
            msg = self.batch_queue.receive_message(visibility_timeout=self.visibility_timeout)
            if msg is not None:
                self.logger.debug('batch: ' + msg.content)
                return msg

            # zaspi na kratly cas
            self.logger.debug('sleeping ' + str(pooling_time) + 's')
            time.sleep(pooling_time)


    def process_batch(self, batch: str) -> bool:
        start_t = datetime.datetime.now(datetime.timezone.utc)

        # vytvorime temporarny adresar pre vstupne subory davky
        with tempfile.TemporaryDirectory() as tmpdirname:

            # vytvorime temporarny adresar pre vysledok spracovania
            with tempfile.TemporaryDirectory() as tmpoutdir:
                self.logger.debug("tmpdirname: " + tmpdirname)
                self.logger.debug("tmpoutdir: " + tmpoutdir)
                prefix = os.path.join(self.batch_folder, batch)
                self.logger.debug('prefix: ' + prefix)
                blob_list = list(self.container.list_blobs(name_starts_with=prefix))

                # download vsetkych suborov do tmpdirname
                for blob in blob_list:
                    local_file_name = blob.name
                    self.logger.debug("blob: " + blob.name)
                    idx = local_file_name.rfind('/')
                    if idx >= 0:
                        local_file_name = local_file_name[idx+1:]
                    download_file_path = os.path.join(tmpdirname, local_file_name)
                    with open(file=download_file_path, mode="wb") as download_file:
                        download_file.write(self.container.download_blob(blob.name).readall())

                # spracovanie vsetkych .csv suborov
                for local_file_name in os.listdir(tmpdirname):
                    if not local_file_name.endswith('.csv'):
                        continue
                    inp_file = os.path.join(tmpdirname, local_file_name)
                    out_file = os.path.join(tmpoutdir, local_file_name[:-4] + '_out.csv')
                    log_file = os.path.join(tmpoutdir, local_file_name[:-4] + '_log.csv')
                    self.logger.debug('inp_file: ' + inp_file)
                    self.logger.debug('out_file: ' + out_file)
                    self.logger.debug('log_file: ' + log_file)

                    # spracovanie suboru
                    try:
                        self.scoring.process_file(inp_file, out_file, log_file)
                        self.logger.debug('process_file: ' + inp_file)
                    except Exception as e:
                        with open(log_file, "a") as log:
                            print(type(e), file=log)
                            print(e, file=log)
                        self.logger.warning(str(type(e)) + ": " + str(e))

                # upload vystupnych suborov do cieloveho foldra
                for local_file_name in os.listdir(tmpoutdir):
                    self.logger.debug('output_file: ' + local_file_name)
                    upload_file_path = os.path.join(tmpoutdir, local_file_name)
                    blob_name = os.path.join(self.output_folder, batch, local_file_name)
                    blob_client = self.container.get_blob_client(blob=blob_name)
                    with open(file=upload_file_path, mode="rb") as data:
                        if blob_client.exists():
                            blob_client.delete_blob()
                        blob_client.upload_blob(data)
                        self.logger.debug('upload result to: ' + blob_name)

                # odmazanie vstupnych suborov davky
                for blob in blob_list:
                    self.container.delete_blob(blob.name)
                    self.logger.debug('delete input blob: ' + blob.name)

        # zaznamenanie ukonceneho spracovania
        end_t = datetime.datetime.now(datetime.timezone.utc)
        out_dict = {'iddavka' : batch,
                    'start_t' : start_t.strftime('%Y-%m-%d %H:%M:%S %Z%z'),
                    'end_t'   : end_t.strftime('%Y-%m-%d %H:%M:%S %Z%z')}
        self.output_queue.send_message(json.dumps(out_dict))
        self.logger.debug('Batch output queued: ' + batch)

        return True

    def run(self):
        fatal_error = False

        while True:
            msg = self.wait_for_batch()

            batch = '???'
            try:

                # mame vybranu davku
                content = msg.content
                json_dict = json.loads(content)
                id_davka = json_dict.get('iddavka')
                batch = 'da_' + str(id_davka)
                # !!!TODO osetrit parsovacie chyby

                # spracovanie jednej davky
                self.logger.info(f'Processing batch: {batch}')
                self.process_batch(batch)
                self.logger.info(f'Batch processed: {batch}')

            except Exception as e:
                self.logger.error(type(e))
                self.logger.error(e)
                fatal_error = True

            finally:
                # odmazanie spravy z fronty
                self.batch_queue.delete_message(msg)
                self.logger.debug('Batch removed from input queue: ' + batch)

            if fatal_error:
                exit(1)


if __name__ == '__main__':

    # --- konfiguracia ---
    STORAGE_ACCOUNT_NAME = os.environ.get('STORAGE_ACCOUNT_NAME', "lmvzs8572858490")
    CONTAINER_NAME = os.environ.get('CONTAINER_NAME', "lmvzsblobstore-container")
    INPUT_FOLDER_NAME = os.environ.get('INPUT_FOLDER_NAME', "batch")
    OUTPUT_FOLDER_NAME = os.environ.get('OUTPUT_FOLDER_NAME', "output")
    INPUT_QUEUE_NAME = os.environ.get('INPUT_QUEUE_NAME', "lmvzs-batch")
    OUTPUT_QUEUE_NAME = os.environ.get('OUTPUT_QUEUE_NAME', "lmvzs-output")
    MODEL_PATH = os.environ.get('MODEL_PATH', "/home/azureuser/cloudfiles/code/Users/bugata/models/2025.02.zip")

    batch_endpoint = BatchEndpoint(model_path=MODEL_PATH,
                                   storage_account=STORAGE_ACCOUNT_NAME,
                                   container=CONTAINER_NAME,
                                   batch_folder=INPUT_FOLDER_NAME,
                                   output_folder=OUTPUT_FOLDER_NAME,
                                   batch_queue=INPUT_QUEUE_NAME,
                                   output_queue=OUTPUT_QUEUE_NAME)

    # spracuje vsetky davky vo vstupnej fronte
    batch_endpoint.run()
