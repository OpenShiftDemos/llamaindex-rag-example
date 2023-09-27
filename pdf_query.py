# Imports
import llama_index
import traceback
from llama_index import set_global_service_context, StorageContext, load_index_from_storage
from llama_index.vector_stores import RedisVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index import ServiceContext
from model_context import get_falcon_tgis_context
from llama_index.prompts.prompts import SimpleInputPrompt

import os, time
import logging
import sys

llama_index.set_global_handler("simple")
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Select Model
#service_context = get_stablelm_context()
#service_context = get_falcon_tgis_context_sentence_window(0.7, 1.03)
service_context = get_falcon_tgis_context(0.7, 1.03)
#service_context = ServiceContext.from_defaults(embed_model="local")

# Load data
redis_hostname = os.getenv('REDIS_SERVER_HOSTNAME', 'localhost') # Get server url from env else default
print("Connecting to Redis at " + redis_hostname)

vector_store = RedisVectorStore(
    index_name="web-console",
    index_prefix="llama",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=False,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)


#query = "How do I get the SSH key for a cluster from Hive?"
query = "How do I install the web terminal?"
response = index.as_query_engine(verbose=True, streaming=True).query(query)
referenced_documents = "\n\nReferenced documents:\n"
for source_node in response.source_nodes:
    #print(source_node.node.metadata['file_name'])
    referenced_documents += source_node.node.metadata['file_name'] + '\n'

print()
print(query)
print(str(response))
print(referenced_documents)
