# Imports
from llama_index import set_global_service_context, StorageContext, load_index_from_storage
from llama_index.vector_stores import RedisVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index import ServiceContext
from model_context import get_falcon_tgis_context
from llama_index.prompts.prompts import SimpleInputPrompt

import os, time
import logging
import sys

system_prompt = """
    - You are a helpful AI assistant and provide the answer for the question based on the given context.
    - You answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, you say "I don't know".
    """
query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")

# Select Model
#service_context = ServiceContext.from_defaults()
service_context = get_falcon_tgis_context(0.7, 1.03)

#set_global_service_context(service_context)

# Load data
vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url="redis://localhost:6379",
    overwrite=True,
)

#storage_context = StorageContext.from_defaults(persist_dir="vector-db")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#index = load_index_from_storage(
#    service_context=service_context,
#    index_id="llama",
#    storage_context=storage_context 
#)

index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

query_engine = index.as_query_engine(
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    #verbose=True,
    #streaming=True,
)


print("Starting front end web app")



# Front end web app
import gradio as gr

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    prompt_box = gr.Textbox(value=system_prompt, label="System Prompt", info="between context and input prompt")
    clear = gr.Button("Clear")
    chat_history = []
    temperature = gr.Slider(minimum=0.1, maximum=2, value=0.7, label="Temperature")
    repetition_penalty = gr.Slider(minimum=0.1, maximum=2, value=1.03, label="Repetition Penalty")

    def user(user_message, history, system_prompt, temperature, repetition_penalty):
        #service_context = ServiceContext.from_defaults()
        #service_context = get_falcon_tgis_context(temperature, repetition_penalty)

        #set_global_service_context(service_context)
        #not sure if we have to re-init query engine
        #index = load_index_from_storage(
        #    service_context=service_context,
        #    storage_context=storage_context 
        #)
        
        if system_prompt == "":
            system_prompt = None
        query_engine = index.as_query_engine(
          system_prompt=system_prompt,
          query_wrapper_prompt=query_wrapper_prompt,
          verbose=True,
          streaming=True,
        )

        # Get response from query engine
        response = query_engine.query(user_message)
        # Append user message and response to chat history
        history.append((user_message, str(response)))
        return gr.update(value=""), history
    msg.submit(user, [msg, chatbot, prompt_box, temperature, repetition_penalty], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    server_port = int(os.getenv("GRADIO_SERVER_PORT", 8055))
    print(f"name is main, server_port for gradio will be {server_port}")
    time.sleep(20)
    print("Launching demo")
    demo.launch(debug=True, server_name="0.0.0.0", server_port=server_port)
