# Document RAG Module

LlamaIndexを使用したドキュメントファイル用RAGシステムのモジュール化実装

## 📁 モジュール構造

```
document_rag/
├── __init__.py          # モジュールエクスポート
├── config.py            # 設定管理 (RAGConfig)
├── logger.py            # ログ管理 (WorkflowLogger)
├── processors.py       # ファイル処理 (TextFileProcessor)
├── managers.py          # インデックス管理 (IndexManager)
├── display.py           # 結果表示 (ResultDisplayer)
├── handlers.py          # 質問応答処理 (QuestionAnswerHandler)
├── pipeline.py          # RAGパイプライン (TextRAGPipeline)
├── workflow.py          # ワークフロー管理 (TextRAGWorkflow)
└── README.md           # このファイル
```

## 🔄 ワークフロー

1. **システム初期化**: LLMモデルと埋め込みモデルの設定
2. **インデックス準備**: 構築または既存インデックスの読み込み
3. **質問応答処理**: 対話モードまたはサンプル実行

## 🎯 主要クラス

### RAGConfig
システム全体の設定を管理
- ファイルパス、モデル設定
- インデックス設定、実行モード
- プロンプトテンプレート、サンプル質問

### TextRAGWorkflow
ワークフロー全体の実行を管理
- 各ステップの順次実行
- エラーハンドリング
- システム状態管理

### TextRAGPipeline
LlamaIndexを使用したRAGパイプライン
- ベクトルインデックス構築
- クエリエンジン設定
- 質問応答処理

## 💡 使用方法

### 基本的な使用
```python
from document_rag import RAGConfig, TextRAGWorkflow

# 設定作成
config = RAGConfig(
    text_path="sample.txt",
    llm_model="gemma:7b",
    interactive=True
)

# ワークフロー実行
workflow = TextRAGWorkflow(config)
workflow.run()
```

### 個別コンポーネントの使用
```python
from document_rag import WorkflowLogger, TextFileProcessor

# ログ機能のみ使用
logger = WorkflowLogger()
logger.log_stage("処理開始", "ファイルを読み込んでいます")

# ファイル処理のみ使用
processor = TextFileProcessor(logger)
documents = processor.load_text_file("sample.txt")
```

## 🔧 カスタマイズ

### 設定のカスタマイズ
```python
config = RAGConfig(
    text_path="custom.txt",
    llm_model="llama2:7b",
    similarity_top_k=5,
    sample_questions=["カスタム質問1", "カスタム質問2"]
)
```

### プロンプトのカスタマイズ
```python
config = RAGConfig(
    text_path="sample.txt",
    japanese_prompt_template="カスタムプロンプト: {context_str}\n質問: {query_str}"
)
```

## 🧪 テスト

各モジュールは独立しているため、個別にテストが可能：

```python
# 設定のテスト
def test_config():
    config = RAGConfig(text_path="test.txt")
    assert config.index_name == "test"

# ログのテスト
def test_logger():
    logger = WorkflowLogger()
    logger.log_info("テストメッセージ")
```

## 🚀 メリット

1. **モジュール性**: 各機能が独立したファイルに分離
2. **再利用性**: 他のプロジェクトで部分的に利用可能
3. **保守性**: 機能追加・修正が容易
4. **テスト性**: 個別コンポーネントのテストが可能
5. **可読性**: 責任が明確で理解しやすい

## 📦 依存関係

- llama_index
- 親ディレクトリの `llamaindex_rag_pipeline`

## 🔄 マイグレーション

元の単一ファイル版からの移行：

```python
# 旧版
from text_rag_llama_index import main
main("sample.txt")

# 新版
from document_rag import RAGConfig, TextRAGWorkflow
config = RAGConfig(text_path="sample.txt")
workflow = TextRAGWorkflow(config)
workflow.run()
``` 