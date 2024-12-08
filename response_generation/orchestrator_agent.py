from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
    def __init__(self, model_name: str, persona_agent, episode_agent, tool_agent, openai_api_key: str):
        self.llm = ChatOpenAI(model=model_name, openai_api_key=openai_api_key)
        self.agents = {
            'persona': persona_agent,
            'episode': episode_agent,
            'tool': tool_agent
        }
    
    def process(self, query: str) -> tuple[str, str]:
        selection = self.llm.predict(
            f"""どのエージェントが以下のクエリに最適か選択してください：
クエリ: {query}
選択肢: persona, episode, tool
回答は選択肢の中から1つのみ選んでください。"""
        )
        
        selected_agent_key = selection.strip().lower()
        selected_agent = self.agents.get(selected_agent_key)
        if selected_agent:
            response = selected_agent.process(query)
            return selected_agent_key, response
        return 'none', ''
