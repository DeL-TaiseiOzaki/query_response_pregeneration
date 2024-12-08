from pathlib import Path

class Config:
    # Model configurations
    MAGPIE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
    OPENAI_MODEL = "gpt-4o-mini"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)