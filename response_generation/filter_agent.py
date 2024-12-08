from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

class QueryFilterAgent(BaseAgent):
    def __init__(self, model_name: str, openai_api_key: str):
        self.llm = ChatOpenAI(model=model_name, openai_api_key=openai_api_key)
    
    def process(self, queries: list[str]) -> list[str]:
        filtered_queries = []
        for query in queries:
            response = self.llm.predict(
                f"""このクエリが自然な質問かどうか判断してください：
クエリ: {query}
回答は'natural'または'unnatural'のみにしてください。"""
            )
            if response.strip().lower() == 'natural':
                filtered_queries.append(query)
        return filtered_queries
