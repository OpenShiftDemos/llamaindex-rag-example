# Imports
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context, ServiceContext
from llama_index.vector_stores import RedisVectorStore
from llama_index.storage.storage_context import StorageContext
from model_context import get_falcon_tgis_context, get_stablelm_context

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url="redis://localhost:6379",
    overwrite=True,
)

# Select Model
#service_context = get_stablelm_context()
service_context = get_falcon_tgis_context(0.7, 1.03)
#service_context = ServiceContext.from_defaults(embed_model="local")

# Load data
filename_fn = lambda filename: {'file_name': filename}

# automatically sets the metadata of each document according to filename_fn
#documents = SimpleDirectoryReader('openshift-docs/webconsole', file_metadata=filename_fn).load_data()
documents = SimpleDirectoryReader('../ops-sop/v4/troubleshoot', file_metadata=filename_fn).load_data()

# Vectorize, index, and store data
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

print("Done indexing!")
