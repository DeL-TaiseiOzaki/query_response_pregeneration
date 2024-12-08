import json
from datetime import datetime
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_score import score
import torch

class DiversePersonaMagpieGenerator:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct", batch_size=4):
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="auto"
        )
        
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 2))
        
        self.batch_size = batch_size
        self.query_params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
        )

    def find_most_similar_query(self, queries, negative_queries):
        if len(queries) < 2:
            return None, 0
        
        # TF-IDFベースの類似度計算
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(queries)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)  # 自分自身との類似度を0に
        
        # 最も類似度が高いペアを探す（ネガティブプロンプトに含まれていないものを優先）
        sorted_pairs = []
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                if similarity_matrix[i, j] > 0:  # 類似度が0より大きい場合のみ考慮
                    sorted_pairs.append((i, j, similarity_matrix[i, j]))
        
        # 類似度で降順ソート
        sorted_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # ネガティブプロンプトに含まれていないクエリの中から最も類似度が高いものを探す
        for i, j, similarity in sorted_pairs:
            query1, query2 = queries[i], queries[j]
            # 両方のクエリがネガティブプロンプトに含まれていない場合
            if query1 not in negative_queries and query2 not in negative_queries:
                print(f"\nTF-IDF Similarity Details:")
                print(f"Query 1: {query1}")
                print(f"Query 2: {query2}")
                print(f"Similarity Score: {similarity:.3f}")
                return query1, similarity  # 最も類似度が高いクエリを返す
        
        return None, 0  # 適切なペアが見つからない場合

    def create_magpie_prompt(self, persona, negative_queries=None):
        base_prompt = f"""<|im_start|>system
あなたは以下のペルソナを持つユーザー専属の優秀なアシスタントです．：

名前: {persona.get('name', '名前未設定')}
年齢: {persona.get('age', '年齢未設定')}
職業: {persona.get('occupation', '職業未設定')}
家族構成: {persona.get('family', '家族構成未設定')}
居住地: {persona.get('residence', '居住地未設定')}
価値観: {persona.get('values', '価値観未設定')}
性格: {persona.get('personality', '性格未設定')}
日課: {persona.get('daily_routine', '日課未設定')}
朝のニュース選択: {persona.get('morning_news_preference', 'ニュース選択未設定')}
夜のニュース選択: {persona.get('evening_news_preference', 'ニュース選択未設定')}
趣味: {persona.get('hobbies', '趣味未設定')}
"""

        if negative_queries and len(negative_queries) > 0:
            base_prompt += """
このユーザーは以下のような質問をすることはありません．：
"""
            for i, query in enumerate(negative_queries, 1):
                base_prompt += f"\n{i}. {query}"

        base_prompt += """

ユーザーからの質問に対して，適切な回答を提供してください．
<|im_end|>
<|im_start|>user
"""
        return base_prompt

    def generate_diverse_queries(self, persona, total_queries=100, step_size=10):
        all_queries = []
        diversity_history = []
        negative_queries = []  # ネガティブプロンプトのリスト
        
        for step in tqdm(range(0, total_queries, step_size)):
            # 蓄積されたネガティブクエリを含むプロンプトを生成
            magpie_prompt = self.create_magpie_prompt(persona, negative_queries)
            prompts = [magpie_prompt] * step_size
            
            # プロンプトを表示
            print("\n" + "="*50 + f"\nStep {step + step_size} Prompt:" + "\n" + "="*50)
            print(magpie_prompt)
            print("="*50 + "\n")
            
            # クエリ生成
            outputs = self.llm.generate(prompts, self.query_params)
            new_queries = []
            for output in outputs:
                try:
                    response = output.outputs[0].text.strip()
                    if response:
                        new_queries.append(response)
                except (IndexError, AttributeError) as e:
                    print(f"Error processing output: {e}")
                    continue
            
            all_queries.extend(new_queries)
            
            # TF-IDFベースで最も類似度の高いクエリを次のステップのネガティブプロンプトとして追加
            negative_query, similarity = self.find_most_similar_query(all_queries, negative_queries)
            if negative_query and negative_query not in negative_queries:
                negative_queries.append(negative_query)
                diversity_history.append(1.0 - similarity)
            
            print(f"\nStep {step + step_size} completed:")
            print(f"Current diversity score (based on TF-IDF): {1.0 - similarity:.3f}")
            print("\nGenerated queries in this step:")
            for i, query in enumerate(new_queries, 1):
                print(f"{i}. {query}")
            print(f"\nTotal negative queries: {len(negative_queries)}")
            if negative_query:
                print(f"New negative query added: {negative_query}")
        
        return all_queries, diversity_history, negative_queries

    def calculate_triple_diversity_metrics(self, queries):
        if len(queries) < 2:
            return 0, 0, 0, []
        
        # 1. Embeddingベースの類似度
        embeddings = self.embedding_model.encode(queries)
        embedding_similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(embedding_similarity_matrix, 0)
        embedding_max_similarities = np.max(embedding_similarity_matrix, axis=1)
        
        # 2. TF-IDFベースの類似度
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(queries)
        tfidf_similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(tfidf_similarity_matrix, 0)
        tfidf_max_similarities = np.max(tfidf_similarity_matrix, axis=1)
        
        # 3. BERTScoreベースの類似度
        bert_similarities = []
        for i, query1 in enumerate(queries):
            other_queries = queries[:i] + queries[i+1:]  # 自分以外のクエリ
            if other_queries:
                _, _, F1 = score([query1] * len(other_queries), other_queries, lang="ja", verbose=False)
                bert_similarities.append(float(F1.max()))
            else:
                bert_similarities.append(0)
        
        avg_embedding_similarity = np.mean(embedding_max_similarities)
        avg_tfidf_similarity = np.mean(tfidf_max_similarities)
        avg_bert_similarity = np.mean(bert_similarities)
        
        return (avg_embedding_similarity, avg_tfidf_similarity, 
                avg_bert_similarity, embedding_max_similarities.tolist())

    def plot_diversity_analysis(self, diversity_history, step_size):
        plt.figure(figsize=(12, 6))
        
        plt.plot(
            range(step_size, len(diversity_history) * step_size + 1, step_size),
            diversity_history,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=8,
            color='#2E86C1',
            label='多様性スコア'
        )
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('生成クエリ数', fontsize=12)
        plt.ylabel('多様性スコア', fontsize=12)
        plt.title('ネガティブプロンプトを用いた多様性分析', fontsize=14, pad=20)
        
        plt.axhline(y=0.3, color='#E74C3C', linestyle='--', alpha=0.5, label='警告閾値 (0.3)')
        
        plt.legend(loc='best')
        plt.ylim(0, 1)
        
        plt.gca().set_facecolor('#F8F9F9')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diversity_analysis_with_negative.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, persona, queries, diversity_history, negative_queries, output_file="diverse_magpie_results.json"):
        data = {
            "persona": persona,
            "queries": queries,
            "negative_queries": negative_queries,
            "diversity_history": [float(d) for d in diversity_history],
            "generated_at": datetime.now().isoformat(),
            "total_queries": len(queries),
            "total_negative_queries": len(negative_queries),
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "analysis_parameters": {
                "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return output_file

def main():
    # サンプルペルソナ
    persona = {
        "name": "樋口 悠斗",
        "age": 25,
        "gender": "男",
        "occupation": "映像クリエイター",
        "family": "妻と小学生の息子1人、娘1人",
        "residence": "北海道旭川市",
        "values": "共同体意識: 集団やコミュニティに所属し、共同体としての利益や幸福を重視する価値観。",
        "personality": "周囲に流される",
        "daily_routine": "平日は、出勤前30分で英語の勉強を行う。",
        "morning_news_preference": "IT・テクノロジーニュース",
        "evening_news_preference": "文化・歴史ニュース",
        "hobbies": "手芸"
    }
    
    try:
        generator = DiversePersonaMagpieGenerator()
        
        print("Starting diverse persona-based Magpie query generation:")
        print(f"- Persona: {persona['name']} ({persona['occupation']})")
        print(f"- Total queries to generate: 100")
        print(f"- Step size for diversity analysis: 10")
        
        queries, diversity_history, negative_queries = generator.generate_diverse_queries(
            persona, total_queries=100, step_size=10
        )
        
        generator.plot_diversity_analysis(diversity_history, step_size=10)
        
        output_file = generator.save_results(
            persona, queries, diversity_history, negative_queries
        )
        
        print(f"\nAnalysis completed and saved to {output_file}")
        print(f"\nGenerated {len(queries)} unique queries")
        print(f"Total negative queries: {len(negative_queries)}")
        
        if queries:
            print("\nSample of generated queries:")
            for i, query in enumerate(queries[:10], 1):
                print(f"{i}. {query}")
            
            print("\nAll negative queries:")
            for i, query in enumerate(negative_queries, 1):
                print(f"{i}. {query}")
            
            print("\nDiversity analysis by steps:")
            for i, div_score in enumerate(diversity_history, 1):
                print(f"Step {i*10}:")
                print(f"  - Diversity score = {div_score:.3f}")
                if div_score < 0.3:
                    print(f"  WARNING: Low diversity detected")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()