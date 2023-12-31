from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.prompts import Prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import LangChainLLM, HuggingFaceLLM
from llama_index import LangchainEmbedding, ServiceContext
from langchain.llms import HuggingFaceTextGenInference
from llama_index.node_parser import SentenceWindowNodeParser
from watsonx_langchain_wrapper import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

import os

def get_stablelm_context():
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """ 

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    # Change default model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    hf_predictor = HuggingFaceLLM(
        context_window=4096, 
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
        model_name="StabilityAI/stablelm-tuned-alpha-3b",
        device_map="cpu",
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16}
    )
    service_context = ServiceContext.from_defaults(chunk_size=1024, llm=hf_predictor, embed_model=embed_model)
    return service_context

def get_falcon_context():
    system_prompt = """# Falcon-7B Instruct
    - You are a helpful AI assistant and provide the answer for the question based on the given context.
    - You answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, you say "I don't know".
    """ 

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")

    # Change default model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    hf_predictor = HuggingFaceLLM(
        context_window=4096, 
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="tiiuae/falcon-7b-instruct",
        model_name="tiiuae/falcon-7b-instruct",
        device_map="auto",
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        # uncomment and add to kwargs if using CUDA to reduce memory usage
        model_kwargs={"trust_remote_code": True} #"torch_dtype": torch.float16
    )
    service_context = ServiceContext.from_defaults(chunk_size=1024, llm=hf_predictor, embed_model=embed_model)
    return service_context

def get_watsonx_predictor(model):
  project_id = os.getenv("WATSON_PROJECT_ID", None)
  creds = {
      "url": "https://us-south.ml.cloud.ibm.com",
      "apikey": os.getenv("WATSON_API_KEY", None)
  }

  params = {
      GenParams.DECODING_METHOD: "greedy",
      GenParams.MIN_NEW_TOKENS: 1,
      GenParams.MAX_NEW_TOKENS: 256
  }

  predictor = LangChainLLM(WatsonxLLM(model=model, credentials=creds, params=params, project_id=project_id))
  embed_model='local:BAAI/bge-base-en'
  service_context = ServiceContext.from_defaults(chunk_size=1024, llm=predictor, 
                                                 #query_wrapper_prompt=query_wrapper_prompt,
                                                 #system_prompt=system_prompt,
                                                 embed_model=embed_model)

  return service_context

def get_tgis_predictor(inference_server_url, temperature, repetition_penalty):
  return LangChainLLM(
        llm=HuggingFaceTextGenInference(
            inference_server_url=inference_server_url,
            max_new_tokens=256,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            server_kwargs={},
        ),
    )

def get_tgis_context_w_extras(temperature, repetition_penalty, system_prompt, query_wrapper_prompt):
  embed_model='local:BAAI/bge-base-en'
  server_url = os.getenv('TGIS_SERVER_URL', 'http://localhost') # Get server url from env else default
  server_port = os.getenv('TGIS_SERVER_PORT', '8049') # Get server port from env else default
  inference_server_url=f"{server_url}:{server_port}/"

  tgis_predictor = get_tgis_predictor(inference_server_url, temperature, repetition_penalty)

  service_context = ServiceContext.from_defaults(chunk_size=1024, llm=tgis_predictor, 
                                                 query_wrapper_prompt=query_wrapper_prompt,
                                                 system_prompt=system_prompt,
                                                 embed_model=embed_model)
  return service_context

def get_falcon_tgis_context(temperature, repetition_penalty):
    system_prompt = """
    - You are a helpful AI assistant and provide the answer for the question based on the given context.
    - You answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, you say "I don't know".
    """ 

    ## This will wrap the default prompts that are internal to llama-index
    #query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")
    query_wrapper_prompt = Prompt("[INST] {query_str} [/INST]")

    print("Changing default model")
    # Change default model
    #embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    embed_model='local:BAAI/bge-base-en'

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
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            server_kwargs={},
        ),
    )

    print("Creating service_context")
    service_context = ServiceContext.from_defaults(chunk_size=1024, llm=tgis_predictor, 
                                                   query_wrapper_prompt=query_wrapper_prompt,
                                                   system_prompt=system_prompt,
                                                   embed_model=embed_model)
    return service_context

def get_falcon_tgis_context_simple(temperature, repetition_penalty):

    print("Changing default model")
    # Change default model
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
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            server_kwargs={},
        ),
    )

    from llama_index import PromptHelper
    prompt_helper = PromptHelper(context_window=2048, num_output=256)

    print("Creating service_context")
    service_context = ServiceContext.from_defaults(chunk_size=512,
                                                   llm=tgis_predictor, 
                                                   context_window=2048,
                                                   prompt_helper=prompt_helper,
                                                   embed_model=embed_model)
    return service_context

def get_falcon_tgis_context_llm_selector(temperature, repetition_penalty):
    system_prompt = """
    - You are a code generation engine.
    - You only respond with JSON objects.
    - You are classifying questions based on provided context.
    - Do not reply with explanation or extraneous infomation.
    """ 

    ## This will wrap the default prompts that are internal to llama-index
    #query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")
    query_wrapper_prompt = Prompt("[INST] {query_str} [/INST] ")

    # Change default model
    #embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    embed_model='local:BAAI/bge-base-en'

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
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            server_kwargs={},
        ),
    )

    print("Creating service_context")
    service_context = ServiceContext.from_defaults(chunk_size=1024, llm=tgis_predictor, 
                                                   #query_wrapper_prompt=query_wrapper_prompt,
                                                   #system_prompt=system_prompt,
                                                   embed_model=embed_model)
    return service_context

def get_falcon_tgis_context_sentence_window(temperature, repetition_penalty):
    #system_prompt = """
    #- You are a helpful AI assistant and provide the answer for the question based on the given context.
    #- You answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, you say "I don't know".
    #""" 

    ## This will wrap the default prompts that are internal to llama-index
    #query_wrapper_prompt = SimpleInputPrompt("### Question\n{query_str}\n### Answer\n")

    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    print("Changing default model")
    # Change default model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

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
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            server_kwargs={},
        ),
    )

    print("Creating service_context")
    #service_context = ServiceContext.from_defaults(chunk_size=1024, llm=tgis_predictor, 
    #                                               query_wrapper_prompt=query_wrapper_prompt,
    #                                               system_prompt=system_prompt,
    #                                               embed_model=embed_model, node_parser=node_parser)
    service_context = ServiceContext.from_defaults(chunk_size=1024, llm=tgis_predictor, 
                                                   embed_model=embed_model, node_parser=node_parser)
    return service_context