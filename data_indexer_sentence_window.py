# Imports
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context, ServiceContext
from llama_index.vector_stores import RedisVectorStore
from llama_index.storage.storage_context import StorageContext
from model_context import get_falcon_tgis_context_sentence_window

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# set up redis as the vector store
vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url="redis://localhost:6379",
    overwrite=True,
)

# Select Model
service_context = get_falcon_tgis_context_sentence_window(0.7, 1.03)
set_global_service_context(service_context)

# Load data
#documents = SimpleDirectoryReader('../ops-sop/v4/troubleshoot').load_data()
documents = SimpleDirectoryReader('private-data').load_data()
#documents = SimpleDirectoryReader('essays').load_data()

# Vectorize, index, and store data
storage_context = StorageContext.from_defaults(vector_store=vector_store)
#index = VectorStoreIndex.from_documents([documents[0]], storage_context=storage_context)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("Done indexing!")
