# Imports
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context, ServiceContext
from llama_index.vector_stores import RedisVectorStore
from llama_index.storage.storage_context import StorageContext
from model_context import get_falcon_tgis_context, get_stablelm_context

import logging, sys, os
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

redis_hostname = os.getenv('REDIS_SERVER_HOSTNAME', 'localhost') # Get server url from env else default
print("Connecting to Redis at " + redis_hostname)

# Select Model
#service_context = get_stablelm_context()
service_context = get_falcon_tgis_context(0.7, 1.03)
#service_context = ServiceContext.from_defaults(embed_model="local")

# Load data
filename_fn = lambda filename: {'file_name': filename}

# Vectorize, index, and store data for each set of documents
#print("Indexing ops documents")
ops_sop_vector_store = RedisVectorStore(
    index_name="ops-sop",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=True,
)

#ops_documents = SimpleDirectoryReader('../ops-sop/v4/troubleshoot', file_metadata=filename_fn).load_data()
#ops_storage_context = StorageContext.from_defaults(vector_store=ops_sop_vector_store)
#ops_index = VectorStoreIndex.from_documents(ops_documents, storage_context=ops_storage_context, service_context=service_context)

#print("Indexing web console documents")
docs_store = RedisVectorStore(
    index_name="openshift-docs",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=True,
)

#webconsole_documents = SimpleDirectoryReader('openshift-docs', file_metadata=filename_fn).load_data()
#webconsole_storage_context = StorageContext.from_defaults(vector_store=docs_store)
#webconsole_index = VectorStoreIndex.from_documents(webconsole_documents, storage_context=webconsole_storage_context, service_context=service_context)

usecase_store = RedisVectorStore(
    index_name="summary-docs",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=True,
)
summary_documents = SimpleDirectoryReader('summary-docs', file_metadata=filename_fn).load_data()
summary_storage_context = StorageContext.from_defaults(vector_store=usecase_store)
summary_index = VectorStoreIndex.from_documents(summary_documents, storage_context=summary_storage_context, service_context=service_context)

print("Done indexing!")
