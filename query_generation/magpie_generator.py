from vllm import LLM, SamplingParams
from .similarity_calculator import SimilarityCalculator
import json

class MagpieGenerator:
    def __init__(self, model_id, batch_size=4):
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="auto"
        )
        self.similarity_calc = SimilarityCalculator()
        self.batch_size = batch_size
        self.query_params = SamplingParams(temperature=0.7, max_tokens=100)

    def create_prompt(self, persona, conversation_history, negative_queries=None):
        prompt = f"""<|im_start|>system
あなたは以下のペルソナを持つユーザー専属のアシスタントです.
：

{json.dumps(persona, ensure_ascii=False, indent=2)}

過去の会話履歴：
{json.dumps(conversation_history, ensure_ascii=False, indent=2)}

ユーザーからの質問に適切に回答して下さい．
"""
        if negative_queries:
            prompt += "\nなおユーザーは以下のような質問をすることはありません：\n"
            prompt += "\n".join(f"- {q}" for q in negative_queries)
        
        prompt += "\n<|im_end|>\n<|im_start|>user"
        return prompt

    def generate_queries(self, persona, conversation_history, total_queries=100, step_size=10):
        queries = []
        negative_queries = []
        
        for step in range(0, total_queries, step_size):
            prompt = self.create_prompt(persona, conversation_history, negative_queries)
            outputs = self.llm.generate([prompt] * step_size, self.query_params)
            
            new_queries = [output.outputs[0].text.strip() for output in outputs if output.outputs]
            queries.extend(new_queries)
            
            similar_query, _ = self.similarity_calc.find_similar_query(queries, negative_queries)
            if similar_query:
                negative_queries.append(similar_query)
        
        return queries, negative_queries