from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

class QueryFilterAgent(BaseAgent):
    def __init__(self, model_name: str):
        self.llm = ChatOpenAI(model=model_name)
    
    def process(self, queries: list[str]) -> list[str]:
        filtered_queries = []
        for query in queries:
            # フィルタリングロジックを実装
            response = self.llm.predict(
                f"""このクエリが自然な質問かどうか判断してください：
                クエリ: {query}
                回答は'natural'または'unnatural'のみにしてください。"""
            )
            if response.strip().lower() == 'natural':
                filtered_queries.append(query)
        return filtered_queries