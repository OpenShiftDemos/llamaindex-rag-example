# Imports
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context, ServiceContext
from llama_index.vector_stores import RedisVectorStore
from llama_index.storage.storage_context import StorageContext
from model_context import get_falcon_tgis_context, get_stablelm_context

import logging, sys, os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

redis_hostname = os.getenv('REDIS_SERVER_HOSTNAME', 'localhost') # Get server url from env else default
print("Connecting to Redis at " + redis_hostname)

ops_sop_vector_store = RedisVectorStore(
    index_name="ops-sop",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=True,
)

web_console_store = RedisVectorStore(
    index_name="web-console",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=True,
)

# Select Model
#service_context = get_stablelm_context()
service_context = get_falcon_tgis_context(0.7, 1.03)
#service_context = ServiceContext.from_defaults(embed_model="local")

# Load data
filename_fn = lambda filename: {'file_name': filename}

# automatically sets the metadata of each document according to filename_fn
webconsole_documents = SimpleDirectoryReader('openshift-docs/webconsole', file_metadata=filename_fn).load_data()
ops_documents = SimpleDirectoryReader('../ops-sop/v4/troubleshoot', file_metadata=filename_fn).load_data()

# Vectorize, index, and store data for each set of documents
print("Indexing ops documents")
ops_storage_context = StorageContext.from_defaults(vector_store=ops_sop_vector_store)
ops_index = VectorStoreIndex.from_documents(ops_documents, storage_context=ops_storage_context, service_context=service_context)

print("Indexing web console documents")
webconsole_storage_context = StorageContext.from_defaults(vector_store=web_console_store)
webconsole_index = VectorStoreIndex.from_documents(webconsole_documents, storage_context=webconsole_storage_context, service_context=service_context)

print("Done indexing!")
