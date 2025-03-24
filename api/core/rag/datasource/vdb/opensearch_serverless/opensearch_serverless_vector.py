import json
import logging
from typing import Any, Optional

from opensearchpy import OpenSearch
from pydantic import BaseModel

from configs import dify_config
from core.rag.datasource.vdb.field import Field
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from models.dataset import Dataset

logger = logging.getLogger(__name__)


class OpenSearchServerlessConfig(BaseModel):
    endpoint: str
    username: Optional[str] = None
    password: Optional[str] = None


class OpenSearchServerlessVector(BaseVector):
    def __init__(self, collection_name: str, config: OpenSearchServerlessConfig):
        super().__init__(collection_name)
        self._client_config = config
        self._client = OpenSearch(hosts=[config.endpoint], http_auth=(config.username, config.password))

    def get_type(self) -> str:
        return VectorType.OPENSEARCH_SERVERLESS

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        self.create_collection(len(embeddings[0]))
        self.add_texts(texts, embeddings)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        actions = []
        for i in range(len(documents)):
            action = {
                "_op_type": "index",
                "_index": self._collection_name.lower(),
                "_source": {
                    Field.CONTENT_KEY.value: documents[i].page_content,
                    Field.VECTOR.value: embeddings[i],
                    Field.METADATA_KEY.value: documents[i].metadata,
                },
            }
            actions.append(action)

        self._client.bulk(actions)

    def text_exists(self, id: str) -> bool:
        try:
            self._client.get(index=self._collection_name.lower(), id=id)
            return True
        except:
            return False

    def delete_by_ids(self, ids: list[str]) -> None:
        for id in ids:
            self._client.delete(index=self._collection_name.lower(), id=id)

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        query = {"query": {"term": {f"{Field.METADATA_KEY.value}.{key}": value}}}
        response = self._client.search(index=self._collection_name.lower(), body=query)
        ids = [hit["_id"] for hit in response["hits"]["hits"]]
        self.delete_by_ids(ids)

    def delete(self) -> None:
        self._client.indices.delete(index=self._collection_name.lower())

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        top_k = kwargs.get("top_k", 4)
        query = {
            "size": top_k,
            "query": {"knn": {Field.VECTOR.value: {"vector": query_vector, "k": top_k}}},
        }
        document_ids_filter = kwargs.get("document_ids_filter")
        if document_ids_filter:
            query["query"] = {"terms": {"metadata.document_id": document_ids_filter}}

        response = self._client.search(index=self._collection_name.lower(), body=query)

        docs = []
        for hit in response["hits"]["hits"]:
            metadata = hit["_source"].get(Field.METADATA_KEY.value, {})
            metadata["score"] = hit["_score"]
            score_threshold = float(kwargs.get("score_threshold") or 0.0)
            if hit["_score"] > score_threshold:
                doc = Document(page_content=hit["_source"].get(Field.CONTENT_KEY.value), metadata=metadata)
                docs.append(doc)

        return docs

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        full_text_query = {"query": {"match": {Field.CONTENT_KEY.value: query}}}
        document_ids_filter = kwargs.get("document_ids_filter")
        if document_ids_filter:
            full_text_query["query"]["terms"] = {"metadata.document_id": document_ids_filter}

        response = self._client.search(index=self._collection_name.lower(), body=full_text_query)

        docs = []
        for hit in response["hits"]["hits"]:
            metadata = hit["_source"].get(Field.METADATA_KEY.value)
            vector = hit["_source"].get(Field.VECTOR.value)
            page_content = hit["_source"].get(Field.CONTENT_KEY.value)
            doc = Document(page_content=page_content, vector=vector, metadata=metadata)
            docs.append(doc)

        return docs

    def create_collection(self, dimension: int):
        lock_name = f"vector_indexing_lock_{self._collection_name.lower()}"
        with redis_client.lock(lock_name, timeout=20):
            collection_exist_cache_key = f"vector_indexing_{self._collection_name.lower()}"
            if redis_client.get(collection_exist_cache_key):
                logger.info(f"Collection {self._collection_name.lower()} already exists.")
                return

            if not self._client.indices.exists(index=self._collection_name.lower()):
                mapping = {
                    "mappings": {
                        "properties": {
                            Field.CONTENT_KEY.value: {"type": "text"},
                            Field.VECTOR.value: {
                                "type": "knn_vector",
                                "dimension": dimension,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "l2",
                                    "engine": "nmslib",
                                    "parameters": {"m": 16, "efconstruction": 200},
                                },
                            },
                            Field.METADATA_KEY.value: {
                                "type": "object",
                                "properties": {
                                    "doc_id": {"type": "keyword"}
                                },
                            },
                        }
                    }
                }
                self._client.indices.create(index=self._collection_name.lower(), body=mapping)

            redis_client.set(collection_exist_cache_key, 1, ex=3600)


class OpenSearchServerlessVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> OpenSearchServerlessVector:
        if dataset.index_struct_dict:
            class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
            collection_name = class_prefix.lower()
        else:
            dataset_id = dataset.id
            collection_name = Dataset.gen_collection_name_by_id(dataset_id).lower()
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.OPENSEARCH_SERVERLESS, collection_name))

        return OpenSearchServerlessVector(
            collection_name=collection_name,
            config=OpenSearchServerlessConfig(
                endpoint=dify_config.OPENSEARCH_SERVERLESS_ENDPOINT,
                username=dify_config.OPENSEARCH_SERVERLESS_USERNAME,
                password=dify_config.OPENSEARCH_SERVERLESS_PASSWORD,
            ),
        )

