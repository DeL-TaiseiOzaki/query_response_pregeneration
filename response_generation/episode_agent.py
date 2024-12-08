from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from config.config import Config
from typing import Dict

class EpisodeAgent(BaseAgent):
    def __init__(self, episode_db: Dict, openai_api_key: str):
        self.episode_db = episode_db
        self.llm = ChatOpenAI(model=Config.OPENAI_MODEL, openai_api_key=openai_api_key)
        
    def process(self, query: str) -> str:
        history = self.episode_db.get("history", [])
        conversation_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        prompt = f"""以下はユーザーとアシスタントの過去の会話履歴です:
{conversation_str}

上記の過去会話を参考に、以下のユーザークエリに答えてください。

ユーザークエリ: {query}
"""
        response = self.llm.predict(prompt)
        return response.strip()
