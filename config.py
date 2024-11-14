from pydantic_settings import BaseSettings
from typing import ClassVar
from langchain_huggingface import HuggingFaceEndpoint
ENV_PATH = '.env'

class Configuration(BaseSettings):
    model_config={
        'case_sensitive':False,
        'env_file': ENV_PATH,
        'extra':'ignore'
    }
    
    TOKEN:str

    
    LLM:ClassVar[HuggingFaceEndpoint]
config=Configuration()
Configuration.LLM=HuggingFaceEndpoint(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1', huggingfacehub_api_token=config.TOKEN, temperature=0.1)