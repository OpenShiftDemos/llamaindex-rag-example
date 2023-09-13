# Imports
from llama_index import set_global_service_context, StorageContext, load_index_from_storage
from llama_index.vector_stores import RedisVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index import ServiceContext
from model_context import get_falcon_tgis_context, get_falcon_tgis_context_sentence_window
from llama_index.prompts.prompts import SimpleInputPrompt

import os, time
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Select Model
#service_context = get_stablelm_context()
#service_context = get_falcon_tgis_context_sentence_window(0.7, 1.03)
service_context = get_falcon_tgis_context(0.7, 1.03)
#service_context = ServiceContext.from_defaults(embed_model="local")

# Load data
redis_hostname = os.getenv('REDIS_SERVER_HOSTNAME', 'localhost') # Get server url from env else default
print("Connecting to Redis at " + redis_hostname)

vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url=f"redis://{redis_hostname}:6379",
    overwrite=False,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

#query_engine = index.as_query_engine(
#    system_prompt=system_prompt,
#    query_wrapper_prompt=query_wrapper_prompt,
#    verbose=True,
#    streaming=True,
#)


#print(index.as_query_engine().query("What is the airspeed velocity of a swallow?"))

#query = "How do I get the SSH key for a cluster from Hive?"
##query = "How do I install the web terminal?"
#response = index.as_query_engine(verbose=True, streaming=True).query(query)
#referenced_documents = "\n\nReferenced documents:\n"
#for source_node in response.source_nodes:
#    #print(source_node.node.metadata['file_name'])
#    referenced_documents += source_node.node.metadata['file_name'] + '\n'
#
#print()
#print(query)
#print(str(response))
#print(referenced_documents)

# Front end web app
print("Starting front end web app")

import gradio as gr

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    #prompt_box = gr.Textbox(value=system_prompt, label="System Prompt", info="between context and input prompt")
    clear = gr.Button("Clear")
    chat_history = []
    temperature = gr.Slider(minimum=0.1, maximum=2, value=0.7, label="Temperature")
    repetition_penalty = gr.Slider(minimum=0.1, maximum=2, value=1.03, label="Repetition Penalty")

    def user(user_message, history, system_prompt, temperature, repetition_penalty):
        
        #query_engine = index.as_query_engine(
        #  verbose=True,
        #  streaming=True,
        #)

        # Get response from query engine
        response = index.as_query_engine(verbose=True, streaming=True).query(user_message)

        #print(response.source_nodes)
        referenced_documents = "\nReferenced documents:\n"
        for source_node in response.source_nodes:
            #print(source_node.node.metadata['file_name'])
            referenced_documents += source_node.node.metadata['file_name'] + '\n'

        # Append user message and response to chat history
        history.append((user_message, str(response) + referenced_documents))
        return gr.update(value=""), history
    #msg.submit(user, [msg, chatbot, prompt_box, temperature, repetition_penalty], [msg, chatbot], queue=False)
    msg.submit(user, [msg, chatbot, temperature, repetition_penalty], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    server_port = int(os.getenv("GRADIO_SERVER_PORT", 8055))
    demo.launch(debug=True, server_name="0.0.0.0", server_port=server_port)
