import llama_index
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores import RedisVectorStore
from llama_index.tools import QueryEngineTool, ToolMetadata
from model_context import get_falcon_tgis_context
import os

llama_index.set_global_handler("simple")

redis_hostname = os.getenv('REDIS_SERVER_HOSTNAME', 'localhost') # Get server url from env else default
print("Connecting to Redis at " + redis_hostname)

vector_store = RedisVectorStore(
    index_name="web-console",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=False,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = get_falcon_tgis_context(0.7, 1.03)
docs_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
docs_engine = docs_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=docs_engine,
        metadata=ToolMetadata(
            name="docs_engine",
            description="Provides information from the OpenShift documentation"
            "Use a detailed plain text question as input to the tool.",
        ),
    )
]

from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.llms import LangChainLLM
from langchain import HuggingFaceTextGenInference

server_url = os.getenv('TGIS_SERVER_URL', 'http://localhost') # Get server url from env else default
server_port = os.getenv('TGIS_SERVER_PORT', '8049') # Get server port from env else default
inference_server_url=f"{server_url}:{server_port}/"

llm = LangChainLLM(
    llm=HuggingFaceTextGenInference(
        inference_server_url=inference_server_url,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.03,
        server_kwargs={},
    ),
)

llm_instruct = OpenAI(model="gpt-3.5-turbo-instruct", api_key="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")

agent = ReActAgent.from_tools(query_engine_tools, llm=llm_instruct, verbose=True)
response = agent.chat("How do I replace the API server certificate?")
