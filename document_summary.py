# Imports
from llama_index import set_global_service_context, ServiceContext
from llama_index.vector_stores import RedisVectorStore
from llama_index import LangchainEmbedding
from llama_index import SimpleDirectoryReader
from llama_index import SummaryIndex
from llama_index.llms import LangChainLLM, HuggingFaceLLM
from langchain import HuggingFaceTextGenInference
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import os, time
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

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

print(f"Getting server environment variables")
server_url = os.getenv('TGIS_SERVER_URL', 'http://localhost') # Get server url from env else default
server_port = os.getenv('TGIS_SERVER_PORT', '8049') # Get server port from env else default
print(f"Initializing TGIS predictor with server_url: {server_url}, server_port: {server_port}")
inference_server_url=f"{server_url}:{server_port}/"
print(f"Inference Service URL: {inference_server_url}")

tgis_predictor = LangChainLLM(
    llm=HuggingFaceTextGenInference(
        inference_server_url=inference_server_url,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.03,
        server_kwargs={},
    ),
)

from llama_index import PromptHelper
prompt_helper = PromptHelper(context_window=1900, num_output=256)

print("Creating service_context")
service_context = ServiceContext.from_defaults(chunk_size=512,
                                               llm=tgis_predictor, 
                                               context_window=2048,
                                               prompt_helper=prompt_helper,
                                               embed_model=embed_model)

# Load data
documents = SimpleDirectoryReader('private-data').load_data()

index = SummaryIndex.from_documents(documents, service_context=service_context)
summary = index.as_query_engine(response_mode="tree_summarize").query("Summarize the text, describing what it might be most useful for")

print(summary.response)

#print()
#print("Node debug stuff")
#nodes = service_context.node_parser.get_nodes_from_documents(documents)
#print(nodes[1].text)