# Imports
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context
from llama_index.vector_stores import RedisVectorStore
from llama_index.storage.storage_context import StorageContext
from model_context import get_falcon_tgis_context

import os, time
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url="redis://localhost:6379",
    overwrite=True,
)

# Select Model
service_context = get_falcon_tgis_context(0.7, 1.03)
set_global_service_context(service_context)

# Load data
documents = SimpleDirectoryReader('../ops-sop/v4/troubleshoot').load_data()

# Vectorize, index, and store data
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

