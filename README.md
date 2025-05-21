# ローカルRAG環境構築プロジェクト

このプロジェクトでは、ローカルでLLMを実行し、RAG（Retrieval Augmented Generation）システムを構築します。


### 環境構築手順

1. **仮想環境をアクティベートする**:
   ```bash
   source venv/bin/activate
   ```

2. **必要なパッケージをインストールする**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ollamaをインストールする**:
   ```bash
   # 以下のガイドを参照
   ./setup_ollama.sh
   ```

4. **LLMの動作確認**:
   ```bash
   # Ollamaサーバーを起動（新しいターミナルウィンドウで）
   ollama serve
   
   # モデルをダウンロード（別のターミナルウィンドウで）
   ollama pull llama3  # または他の希望するモデル
   
   # 動作確認スクリプトを実行
   python scripts/check_llm.py
   ```

## プロジェクト構造

```
local-rag/
├── data/           # ドキュメントデータ（PDF等）を格納
├── models/         # モデル設定やチェックポイントを格納
├── notebooks/      # 実験用Jupyterノートブック
├── scripts/        # ユーティリティスクリプト
│   └── check_llm.py # LLM動作確認スクリプト
├── src/            # ソースコード
├── venv/           # Python仮想環境
├── requirements.txt # 依存パッケージリスト
├── setup_ollama.sh # Ollamaセットアップガイド
└── README.md       # プロジェクト説明
```