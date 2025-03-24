from pydantic import Field
from pydantic_settings import BaseSettings


class OpenSearchServerlessConfig(BaseSettings):
    """
    Configuration settings for OpenSearch Serverless
    """

    OPENSEARCH_SERVERLESS_ENDPOINT: str = Field(
        description="URL of the OpenSearch Serverless service (e.g., 'https://localhost:9200')",
        default="https://localhost:9200",
    )

    OPENSEARCH_SERVERLESS_USERNAME: str = Field(description="Username for authenticating with OpenSearch Serverless", default="admin")

    OPENSEARCH_SERVERLESS_PASSWORD: str = Field(description="Password for authenticating with OpenSearch Serverless", default="admin")

