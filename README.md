# マルチエージェント会話生成システム

## 概要
このシステムは、ユーザーのペルソナと過去の会話履歴に基づいて、自然な会話を生成するマルチエージェントフレームワークです。Magpieを使用してクエリを生成し、複数の専門エージェントが協調して適切な応答を生成します。

## システム構成
```
agent_system/
├── config/
│   └── config.py          # システム設定
├── query_generation/
│   ├── magpie_generator.py    # クエリ生成
│   └── similarity_calculator.py # 類似度計算
├── agent_framework/
│   ├── filter_agent.py        # クエリフィルタリング
│   ├── orchestrator_agent.py  # エージェント制御
│   ├── persona_agent.py       # ペルソナベース応答
│   ├── episode_agent.py       # エピソードベース応答
│   └── tool_agent.py          # ツールベース応答
├── data/
│   └── persona.json       # 入力データ
├── outputs/               # 生成結果の出力先
├── main.py               # メインスクリプト
└── run_agent_system.sh   # 実行スクリプト
```

## 必要条件
- Python 3.8以上
- 必要なPythonパッケージ:
  ```
  vllm
  langchain
  langchain-openai
  scikit-learn
  numpy
  ```

## インストール
```bash
# リポジトリのクローン
git clone [repository-url]
cd agent_system

# 必要なパッケージのインストール
pip install -r requirements.txt

# 実行権限の付与
chmod +x run_agent_system.sh
```

## 使用方法

### 入力データの準備
`data/persona.json`に以下の形式でデータを配置します：
```json
{
  "name": "名前",
  "age": 25,
  "gender": "性別",
  "occupation": "職業",
  ... // その他のペルソナ情報
  "history": [
    [
      {
        "role": "user",
        "content": "ユーザーの発話"
      },
      {
        "role": "assistant",
        "content": "アシスタントの応答"
      }
    ]
    // 複数の会話セッション
  ]
}
```

### システムの実行
```bash
./run_agent_system.sh
```

### 出力
生成結果は`outputs/`ディレクトリに以下の形式で保存されます：
```json
{
  "persona": {
    // 入力されたペルソナ情報
  },
  "generated_queries": [
    {
      "query": "生成されたクエリ",
      "agent": "担当エージェント",
      "response": "生成された応答"
    }
  ]
}
```

## 処理フロー
1. Magpieによるクエリ生成（10件ずつ、ネガティブプロンプト使用）
2. クエリフィルタエージェントによる不自然なクエリの除外
3. オーケストレーションエージェントによる適切なエージェントの選択
4. 各専門エージェントによる応答生成
5. 結果のJSON形式での保存

## 主要コンポーネント
- **MagpieGenerator**: ユーザーの特性に基づくクエリ生成
- **QueryFilterAgent**: 不自然なクエリの除外
- **OrchestratorAgent**: 適切なエージェントの選択
- **専門エージェント**:
  - PersonaAgent: ペルソナベースの応答生成
  - EpisodeAgent: 過去の会話に基づく応答生成
  - ToolAgent: 外部ツールを使用した応答生成
