# Imports
import llama_index
import traceback
from llama_index.vector_stores import RedisVectorStore
from llama_index import VectorStoreIndex
from model_context import get_falcon_tgis_context, get_falcon_tgis_context_llm_selector
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.callbacks import CallbackManager, TokenCountingHandler

import os, time
import logging
import sys

llama_index.set_global_handler("simple")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

token_counter = TokenCountingHandler()
callback_manager = CallbackManager([token_counter])

# Select Model
service_context = get_falcon_tgis_context(0.7, 1.03)

llm_selector_context = get_falcon_tgis_context_llm_selector(0.7, 1.03)

# Load data
redis_hostname = os.getenv('REDIS_SERVER_HOSTNAME', 'localhost') # Get server url from env else default
print("Connecting to Redis at " + redis_hostname)

# https://gpt-index.readthedocs.io/en/stable/examples/query_engine/RouterQueryEngine.html
# attempt to use a router query setup

print("Setting up vector stores")
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

print("Setting up vector indices")
ops_index = VectorStoreIndex.from_vector_store(ops_sop_vector_store, service_context=service_context)
web_console_index = VectorStoreIndex.from_vector_store(web_console_store, service_context=service_context)

print("Setting up query engines")
ops_query_engine = ops_index.as_query_engine(
    verbose=True,
    streaming=True,
)

web_console_engine = web_console_index.as_query_engine(
    verbose=True,
    streaming=True,
)

print("Setting up tools")
os_query_tool = QueryEngineTool.from_defaults(
    query_engine=ops_query_engine,
    description="Documents related to SRE and operations questions about troubleshooting managed OpenShift clusters.",
)

web_console_query_tool = QueryEngineTool.from_defaults(
    query_engine=web_console_engine,
    description="User and administrator documentation related to the OpenShift web console and its configuration.",
)

print("Setting up router")
query_engine = RouterQueryEngine(
    service_context=service_context,
    selector=LLMSingleSelector.from_defaults(service_context=llm_selector_context),
    query_engine_tools=[
        os_query_tool,
        web_console_query_tool,
    ],
)

#query = input("What's your query? ")
query = "I am an OpenShift administrator and I would like to know how to install the web terminal."

try:
    response = query_engine.query(query)
    referenced_documents = "\n\nReferenced documents:\n"
    for source_node in response.source_nodes:
        #print(source_node.node.metadata['file_name'])
        referenced_documents += source_node.node.metadata['file_name'] + '\n'

    #print()
    #print(query)
    #print()
    #print("all prompts (including intermediary")
    #print()
    #for x in range(len(token_counter.llm_token_counts)):
    #    print("prompt ", x)
    #    print("prompt: ", token_counter.llm_token_counts[x].prompt)
    print("response")
    print(str(response))
    print(referenced_documents)
except Exception:
    traceback.print_exc()
    #for x in range(len(token_counter.llm_token_counts)):
    #    print("prompt ", x)
    #    print("prompt: \n", token_counter.llm_token_counts[x].prompt)
    #    print()
    #    print("completion: \n", token_counter.llm_token_counts[x].completion)
