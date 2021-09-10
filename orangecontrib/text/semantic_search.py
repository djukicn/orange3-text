import json
import base64
import zlib
import sys
import re
from typing import Any, Optional, List, Optional, Tuple, Callable

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data


MAX_PACKAGE_SIZE = 500000


class SemanticSearch:

    def __init__(self) -> None:
        self._server_communicator = _ServerCommunicator(
            model_name='semantic-search',
            max_parallel_requests=100,
            server_url='https://apiv2.garaza.io',
            embedder_type='text',
        )

    def __call__(
        self, texts: List[str], queries: List[str],
        progress_callback: Optional[Callable] = None
    ) -> List[Optional[List[Tuple[Tuple[int, int], float]]]]:

        chunks = list()
        chunk = list()
        chunk_size = 0
        skipped = list()
        queries_enc = base64.b64encode(
            zlib.compress(
                json.dumps(queries).encode('utf-8', 'replace'),
                level=-1
            )
        ).decode('utf-8', 'replace')

        for i, text in enumerate(texts):
            encoded = base64.b64encode(zlib.compress(
                text.encode('utf-8', 'replace'), level=-1)
            ).decode('utf-8', 'replace')
            size = sys.getsizeof(encoded)
            if size > MAX_PACKAGE_SIZE:
                skipped.append(i)
                continue
            if chunk_size + size > MAX_PACKAGE_SIZE:
                chunks.append([chunk, queries_enc])
                chunk = list()
                chunk_size = 0
            chunk.append(encoded)
            chunk_size += size

        chunks.append([chunk, queries_enc])

        result = self._server_communicator.embedd_data(
            chunks, processed_callback=progress_callback,
        )[0]
        if result is None:
            return [None] * len(texts)

        results = list()
        idx = 0
        for i in range(len(texts)):
            if i in skipped:
                results.append(None)
            else:
                results.append(result[idx])
                idx += 1

        return results

    def set_cancelled(self):
        if hasattr(self, '_server_communicator'):
            self._server_communicator.set_cancelled()

    def clear_cache(self):
        """Clears embedder cache"""
        if self._server_communicator:
            self._server_communicator.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, ex_type, value, traceback):
        self.set_cancelled()

    def __del__(self):
        self.__exit__(None, None, None)


class _ServerCommunicator(ServerEmbedderCommunicator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = 'application/json'

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
        return json.dumps(data_instance).encode('utf-8', 'replace')
