#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM動作確認スクリプト
=====================
Ollamaを使用してローカルLLMの動作を確認するための簡単なスクリプトです。
"""

from langchain_ollama import OllamaLLM
import time

def test_ollama_connection(model_name="gemma:7b",):
    """
    Ollamaサーバーに接続し、指定したモデルで簡単なプロンプトに応答させる
    
    Args:
        model_name: 使用するモデル名 (デフォルト: "llama3")
    
    Returns:
        なし。結果をコンソールに出力します。
    """
    print(f"🔍 {model_name}モデルの接続テスト中...")
    
    try:
        # Ollamaモデルのインスタンス化
        llm = OllamaLLM(model=model_name)
        
        # テスト用のシンプルなプロンプト
        prompt = "Hello! Can you introduce yourself in one sentence?"
        
        # 実行開始時間
        start_time = time.time()
        
        # モデルに問い合わせ
        response = llm.invoke(prompt)
        
        # 実行終了時間
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n✅ LLMのテスト成功!")
        print(f"⏱️  応答時間: {execution_time:.2f}秒")
        print("\n📝 応答文:")
        print(f"{response}")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        print("\n考えられる原因:")
        print("1. Ollamaがインストールされていないか、実行されていない")
        print("2. 指定されたモデルがダウンロードされていない")
        print("3. ネットワーク接続に問題がある")
        print("\n解決方法:")
        print("1. Ollamaをインストールして実行: `ollama serve`")
        print(f"2. モデルをダウンロード: `ollama pull {model_name}`")

if __name__ == "__main__":
    # デフォルトのモデルでテスト実行
    test_ollama_connection()
    
    # 以下のコメントを解除して他のモデルをテスト
    # test_ollama_connection("mistral")
    # test_ollama_connection("llama2")
