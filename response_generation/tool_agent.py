from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from config.config import Config
import requests

class ToolAgent(BaseAgent):
    def __init__(self, api_key: str, cse_id: str, openai_api_key: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.llm = ChatOpenAI(model=Config.OPENAI_MODEL, openai_api_key=openai_api_key)
        
    def google_search(self, query: str) -> str:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "items" in data:
            results = data["items"][:3]
            snippets = [item.get("snippet", "") for item in results if "snippet" in item]
            return "\n".join(snippets)
        return "検索結果が見つかりません。"
    
    def process(self, query: str) -> str:
        search_results = self.google_search(query)
        prompt = f"""以下はGoogle検索結果の一部です:
{search_results}

上記検索結果を参考に、ユーザークエリ「{query}」に適切に回答してください。
"""
        response = self.llm.predict(prompt)
        return response.strip()
