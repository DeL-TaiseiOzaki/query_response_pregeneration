from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from .base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
    def __init__(self, model_name: str, persona_agent, episode_agent, tool_agent):
        self.llm = ChatOpenAI(model=model_name)
        self.agents = {
            'persona': persona_agent,
            'episode': episode_agent,
            'tool': tool_agent
        }
    
    def process(self, query: str) -> tuple[str, str]:
        # エージェント選択ロジックを実装
        selection = self.llm.predict(
            f"""どのエージェントが以下のクエリに最適か選択してください：
            クエリ: {query}
            選択肢: persona, episode, tool
            回答は選択肢の中から1つのみ選んでください。"""
        )
        
        selected_agent = self.agents.get(selection.strip().lower())
        if selected_agent:
            response = selected_agent.process(query)
            return selection, response
        return 'none', ''