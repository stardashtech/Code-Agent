import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import uuid

# Assuming the VectorStoreManager class is in app.services.vector_store_manager
from app.services.vector_store_manager import VectorStoreManager

# Mock data
FAKE_QDRANT_HOST = "http://localhost:6333"
FAKE_EMBEDDING_DIM = 10 # Use a small dimension for tests
FAKE_QUERY = "test query"
FAKE_CODE = "def hello(): print('world')"
FAKE_METADATA = {"file_path": "test.py", "language": "python"}
FAKE_EMBEDDING = [0.1] * FAKE_EMBEDDING_DIM
FAKE_INTERACTION_DATA = {
    'query': FAKE_QUERY,
    'status': 'success',
    'code_snippets': [],
    'fix_status': 'success',
    'has_fix': True,
    'explanation': 'test explanation'
}

# --- Fixtures --- #

@pytest.fixture
def mock_qdrant_client(mocker):
    """Fixture for a mocked QdrantClient instance."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[]) # Default: no collections
    mock_client.search = MagicMock() # Mock the search method
    mock_client.upsert = MagicMock() # Mock the upsert method
    mock_client.create_collection = MagicMock()
    mock_client.delete_collection = MagicMock()
    # Patch the QdrantClient class within the vector_store_manager module
    mocker.patch('app.services.vector_store_manager.QdrantClient', return_value=mock_client)
    return mock_client

@pytest.fixture
def mock_embedding_func():
    """Fixture for a mocked async embedding function."""
    # Use AsyncMock for async functions
    return AsyncMock(return_value=FAKE_EMBEDDING)

@pytest.fixture
def vector_store_manager_instance(mock_qdrant_client, mock_embedding_func):
    """Fixture to create a VectorStoreManager instance with mocks."""
    # We pass the actual async mock function here
    manager = VectorStoreManager(
        qdrant_host=FAKE_QDRANT_HOST, 
        embedding_dimension=FAKE_EMBEDDING_DIM, 
        embedding_func=mock_embedding_func
    )
    # Replace the client instance created within __init__ with our main mock
    manager.qdrant_client = mock_qdrant_client 
    return manager

# --- Test Cases --- #

@pytest.mark.asyncio
async def test_init_ensures_collection_exists(mock_qdrant_client, mock_embedding_func):
    """Test that __init__ calls _ensure_collection_exists, which checks/creates the collection."""
    # Reset mock before creating instance
    mock_qdrant_client.reset_mock()
    mock_qdrant_client.get_collections.return_value = MagicMock(collections=[]) # Simulate collection doesn't exist

    VectorStoreManager(
        qdrant_host=FAKE_QDRANT_HOST, 
        embedding_dimension=FAKE_EMBEDDING_DIM, 
        embedding_func=mock_embedding_func
    )

    mock_qdrant_client.get_collections.assert_called_once()
    # Check that create_collection was called with the correct parameters
    mock_qdrant_client.create_collection.assert_called_once_with(
        collection_name=f"code_embeddings_{FAKE_EMBEDDING_DIM}",
        vectors_config=ANY # Check VectorParams details if necessary
    )

@pytest.mark.asyncio
async def test_init_uses_existing_collection(mock_qdrant_client, mock_embedding_func):
    """Test that __init__ uses an existing collection and doesn't recreate it."""
    mock_qdrant_client.reset_mock()
    # Simulate collection already exists
    existing_collection_name = f"code_embeddings_{FAKE_EMBEDDING_DIM}"
    mock_collection = MagicMock()
    mock_collection.name = existing_collection_name
    mock_qdrant_client.get_collections.return_value = MagicMock(collections=[mock_collection])

    VectorStoreManager(
        qdrant_host=FAKE_QDRANT_HOST, 
        embedding_dimension=FAKE_EMBEDDING_DIM, 
        embedding_func=mock_embedding_func
    )

    mock_qdrant_client.get_collections.assert_called_once()
    mock_qdrant_client.create_collection.assert_not_called()

@pytest.mark.asyncio
async def test_search_code_success(vector_store_manager_instance, mock_embedding_func, mock_qdrant_client):
    """Test successful code search."""
    # Configure mock search results
    mock_point = MagicMock()
    mock_point.id = str(uuid.uuid4())
    mock_point.score = 0.95
    mock_point.payload = {'code': FAKE_CODE, 'file_path': 'test.py', 'language': 'python', 'timestamp': 'ts'}
    mock_qdrant_client.search.return_value = [mock_point]

    results = await vector_store_manager_instance.search_code(FAKE_QUERY)

    mock_embedding_func.assert_awaited_once_with(FAKE_QUERY)
    mock_qdrant_client.search.assert_called_once_with(
        collection_name=vector_store_manager_instance.collection_name,
        query_vector=FAKE_EMBEDDING,
        limit=5
    )
    assert len(results) == 1
    assert results[0]['code'] == FAKE_CODE
    assert results[0]['similarity'] == 0.95
    assert results[0]['id'] == mock_point.id

@pytest.mark.asyncio
async def test_search_code_no_results(vector_store_manager_instance, mock_embedding_func, mock_qdrant_client):
    """Test code search when Qdrant returns no results."""
    mock_qdrant_client.search.return_value = [] # Simulate empty results

    results = await vector_store_manager_instance.search_code(FAKE_QUERY)

    mock_embedding_func.assert_awaited_once_with(FAKE_QUERY)
    mock_qdrant_client.search.assert_called_once()
    assert results == []

@pytest.mark.asyncio
async def test_search_code_embedding_error(vector_store_manager_instance, mock_embedding_func):
    """Test code search when embedding generation fails."""
    mock_embedding_func.side_effect = Exception("Embedding failed")

    results = await vector_store_manager_instance.search_code(FAKE_QUERY)

    mock_embedding_func.assert_awaited_once_with(FAKE_QUERY)
    assert results == [] # Should return empty list on error

@pytest.mark.asyncio
async def test_store_code_success(vector_store_manager_instance, mock_embedding_func, mock_qdrant_client):
    """Test successfully storing code."""
    point_id = await vector_store_manager_instance.store_code(FAKE_METADATA['file_path'], FAKE_CODE, FAKE_METADATA['language'])

    mock_embedding_func.assert_awaited_once_with(FAKE_CODE)
    mock_qdrant_client.upsert.assert_called_once()
    # Check payload details if needed, using ANY for dynamic values like id/timestamp
    call_args = mock_qdrant_client.upsert.call_args
    assert call_args[1]['collection_name'] == vector_store_manager_instance.collection_name
    assert len(call_args[1]['points']) == 1
    payload = call_args[1]['points'][0].payload
    assert payload['code'] == FAKE_CODE
    assert payload['file_path'] == FAKE_METADATA['file_path']
    assert payload['language'] == FAKE_METADATA['language']
    assert payload['type'] == 'code_master'
    assert isinstance(point_id, str)

@pytest.mark.asyncio
async def test_store_code_failure(vector_store_manager_instance, mock_embedding_func, mock_qdrant_client):
    """Test code storage failure during upsert."""
    mock_qdrant_client.upsert.side_effect = Exception("Upsert failed")

    point_id = await vector_store_manager_instance.store_code(FAKE_METADATA['file_path'], FAKE_CODE, FAKE_METADATA['language'])

    mock_embedding_func.assert_awaited_once_with(FAKE_CODE)
    mock_qdrant_client.upsert.assert_called_once()
    assert point_id is None # Should return None on failure

@pytest.mark.asyncio
async def test_save_interaction_success(vector_store_manager_instance, mock_embedding_func, mock_qdrant_client):
    """Test successfully saving interaction data."""
    point_id = await vector_store_manager_instance.save_interaction(FAKE_INTERACTION_DATA)

    mock_embedding_func.assert_awaited_once() # Check that it was called
    mock_qdrant_client.upsert.assert_called_once()
    call_args = mock_qdrant_client.upsert.call_args
    payload = call_args[1]['points'][0].payload
    assert payload['query'] == FAKE_INTERACTION_DATA['query']
    assert payload['type'] == 'interaction'
    assert payload['status'] == FAKE_INTERACTION_DATA['status']
    assert payload['fix_status'] == FAKE_INTERACTION_DATA['fix_status']
    assert isinstance(point_id, str)

# TODO: Add more tests:
# - test_index_code success and failure
# - test_save_interaction failure
# - test_search_code with Qdrant client error
# - test _delete_collection
