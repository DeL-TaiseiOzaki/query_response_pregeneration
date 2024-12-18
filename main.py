import argparse
import json
import sys
from pathlib import Path
from config.config import Config
from query_generation.magpie_generator import MagpieGenerator
from response_generation.filter_agent import QueryFilterAgent
from response_generation.orchestrator_agent import OrchestratorAgent
from response_generation.persona_agent import PersonaAgent
from response_generation.episode_agent import EpisodeAgent
from response_generation.tool_agent import ToolAgent
import os
from dotenv import load_dotenv

# .envファイル読み込み
load_dotenv()

def load_persona_data(file_path: str) -> tuple[dict, list]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        persona = {k: v for k, v in data.items() if k != 'history'}
        conversation_history = data.get('history', [])
        
        return persona, conversation_history
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON")
        sys.exit(1)

def process_queries(persona: dict, conversation_history: list, output_path: str):
    try:
        # 環境変数からAPIキーを取得
        openai_api_key = os.getenv("OPENAI_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")

        magpie = MagpieGenerator(Config.MAGPIE_MODEL_ID)
        filter_agent = QueryFilterAgent(Config.OPENAI_MODEL, openai_api_key=openai_api_key)
        
        persona_agent = PersonaAgent(persona, openai_api_key=openai_api_key)
        episode_agent = EpisodeAgent({"history": conversation_history}, openai_api_key=openai_api_key)
        tool_agent = ToolAgent(api_key=google_api_key, cse_id=google_cse_id, openai_api_key=openai_api_key)

        orchestrator = OrchestratorAgent(
            Config.OPENAI_MODEL,
            persona_agent,
            episode_agent,
            tool_agent,
            openai_api_key=openai_api_key
        )
        
        print("Generating queries...")
        queries, _ = magpie.generate_queries(persona, conversation_history)
        
        print("Filtering queries...")
        filtered_queries = filter_agent.process(queries)
        
        print("Processing queries through orchestrator...")
        results = []
        for i, query in enumerate(filtered_queries, 1):
            print(f"Processing query {i}/{len(filtered_queries)}")
            agent_type, response = orchestrator.process(query)
            results.append({
                "query": query,
                "agent": agent_type,
                "response": response
            })
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "persona": persona,
                "generated_queries": results
            }, ensure_ascii=False, indent=2, fp=f)
            
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Process persona data and generate queries')
    parser.add_argument('input_file', help='Path to the input JSON file containing persona and conversation history')
    parser.add_argument('output_file', help='Path to save the output JSON file')
    
    args = parser.parse_args()
    
    persona, conversation_history = load_persona_data(args.input_file)
    process_queries(persona, conversation_history, args.output_file)

if __name__ == "__main__":
    main()
