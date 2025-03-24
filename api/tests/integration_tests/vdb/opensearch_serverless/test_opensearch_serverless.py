from core.rag.datasource.vdb.opensearch_serverless.opensearch_serverless_vector import (
    OpenSearchServerlessConfig,
    OpenSearchServerlessVector,
)
from tests.integration_tests.vdb.test_vector_store import (
    AbstractVectorTest,
    setup_mock_redis,
)


class OpenSearchServerlessVectorTest(AbstractVectorTest):
    def __init__(self):
        super().__init__()
        self.vector = OpenSearchServerlessVector(
            collection_name=self.collection_name,
            config=OpenSearchServerlessConfig(
                endpoint="https://localhost:9200",
                username="admin",
                password="admin",
            ),
        )

    def get_ids_by_metadata_field(self):
        ids = self.vector.get_ids_by_metadata_field(key="document_id", value=self.example_doc_id)
        assert len(ids) == 1


def test_opensearch_serverless_vector(setup_mock_redis):
    OpenSearchServerlessVectorTest().run_all_tests()

