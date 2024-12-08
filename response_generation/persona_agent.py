from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from config.config import Config

class PersonaAgent(BaseAgent):
    def __init__(self, persona: dict, openai_api_key: str):
        self.persona = persona
        self.llm = ChatOpenAI(model=Config.OPENAI_MODEL, openai_api_key=openai_api_key)
        
    def process(self, query: str) -> str:
        prompt = f"""以下はユーザーのペルソナ情報です:
{self.persona}

ユーザーは上記のペルソナを持っています。以下のユーザーからの質問に対して、ペルソナに即した回答をしてください。

ユーザークエリ: {query}
"""
        response = self.llm.predict(prompt)
        return response.strip()
