import pytest
from types import SimpleNamespace


ingq = __import__("scripts.ingest.qdrant", fromlist=["*"])


class FakeClient:
    def __init__(self, *, collection_exists=True, vectors=None, sparse_vectors=None, payload_schema=None):
        self.collection_exists = collection_exists
        self.vectors = vectors or {}
        self.sparse_vectors = sparse_vectors or {}
        self.payload_schema = payload_schema or {}
        self.create_calls = []
        self.update_calls = []
        self.payload_index_calls = []
        self.get_calls = 0

    def get_collection(self, name):
        self.get_calls += 1
        if not self.collection_exists:
            raise RuntimeError("not found")
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=self.vectors,
                    sparse_vectors=self.sparse_vectors,
                )
            ),
            payload_schema=self.payload_schema,
        )

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None, hnsw_config=None, quantization_config=None):
        self.create_calls.append(
            {
                "collection_name": collection_name,
                "vectors_config": vectors_config,
                "sparse_vectors_config": sparse_vectors_config,
                "hnsw_config": hnsw_config,
                "quantization_config": quantization_config,
            }
        )
        self.collection_exists = True
        self.vectors = dict(vectors_config)
        self.sparse_vectors = dict(sparse_vectors_config or {})

    def update_collection(self, collection_name, vectors_config):
        self.update_calls.append(
            {
                "collection_name": collection_name,
                "vectors_config": vectors_config,
            }
        )
        self.vectors.update(vectors_config)

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.payload_index_calls.append(
            {"collection_name": collection_name, "field_name": field_name}
        )
        self.payload_schema[field_name] = field_schema


def test_schema_mode_validate_errors_on_missing_vectors(monkeypatch):
    monkeypatch.setenv("PATTERN_VECTORS", "1")
    monkeypatch.setattr(ingq, "LEX_SPARSE_MODE", False)

    existing_vectors = {
        "code": object(),
        ingq.LEX_VECTOR_NAME: object(),
    }
    payload_schema = {field: object() for field in ingq.PAYLOAD_INDEX_FIELDS}
    client = FakeClient(
        collection_exists=True,
        vectors=existing_vectors,
        payload_schema=payload_schema,
    )

    with pytest.raises(RuntimeError, match="schema mismatch"):
        ingq.ensure_collection(
            client,
            "test-collection",
            dim=8,
            vector_name="code",
            schema_mode="validate",
        )

    assert client.update_calls == []
    assert client.create_calls == []


def test_schema_mode_migrate_adds_missing_vectors_and_indexes(monkeypatch):
    monkeypatch.setenv("PATTERN_VECTORS", "1")
    monkeypatch.setattr(ingq, "LEX_SPARSE_MODE", False)

    existing_vectors = {
        "code": object(),
        ingq.LEX_VECTOR_NAME: object(),
    }
    client = FakeClient(
        collection_exists=True,
        vectors=existing_vectors,
        payload_schema={},
    )

    ingq.ensure_collection(
        client,
        "test-collection",
        dim=8,
        vector_name="code",
        schema_mode="migrate",
    )

    assert client.create_calls == []
    assert client.update_calls
    updated_vectors = client.update_calls[0]["vectors_config"]
    assert ingq.PATTERN_VECTOR_NAME in updated_vectors
    assert any(
        c["field_name"] == "metadata.language" for c in client.payload_index_calls
    )


def test_schema_mode_create_creates_collection_only(monkeypatch):
    monkeypatch.setenv("PATTERN_VECTORS", "0")
    monkeypatch.setattr(ingq, "LEX_SPARSE_MODE", False)

    client = FakeClient(collection_exists=False)

    ingq.ensure_collection(
        client,
        "test-collection",
        dim=8,
        vector_name="code",
        schema_mode="create",
    )

    assert len(client.create_calls) == 1
    assert client.update_calls == []
    assert any(
        c["field_name"] == "metadata.language" for c in client.payload_index_calls
    )
