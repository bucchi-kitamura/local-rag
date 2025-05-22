#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LlamaIndex RAGパイプラインモジュール
===============================
LlamaIndexを使用したPDF読み込み、インデックス作成、質問応答パイプライン
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llamaindex_document_loader import LlamaIndexDocumentProcessor


class LlamaIndexRAGPipeline:
    """LlamaIndexを使用したRAGパイプライン"""

    def __init__(
        self,
        llm_model: str = "gemma:7b",
        embed_model: str = "intfloat/multilingual-e5-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        models_dir: str = "models",
    ):
        """
        LlamaIndexRAGPipelineのコンストラクタ
        
        Args:
            llm_model: 使用するOllamaのLLMモデル名
            embed_model: 使用する埋め込みモデル名
            chunk_size: チャンクサイズ（文字数）
            chunk_overlap: チャンク間のオーバーラップ（文字数）
            models_dir: モデルディレクトリのパス
        """
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.models_dir = models_dir

        # LLMの設定（日本語レスポンスを強制）
        self.llm = Ollama(
            model=llm_model, 
            request_timeout=300.0,
            system_prompt="あなたは日本語で回答するアシスタントです。必ず日本語で回答してください。英語での回答は禁止です。"
        )
        
        # 埋め込みモデルの設定
        self.embedding = HuggingFaceEmbedding(model_name=embed_model)
        
        # グローバル設定を行う
        Settings.llm = self.llm
        Settings.embed_model = self.embedding
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # コンポーネントの初期化
        self.document_processor = LlamaIndexDocumentProcessor()
        self.index = None
        self.query_engine = None
        
        # 状態管理
        self.is_index_built = False

    def build_index_from_pdf(
        self,
        pdf_path: str,
        save_index: bool = True,
        index_name: Optional[str] = None,
    ) -> None:
        """
        PDFからベクトルインデックスを構築する
        
        Args:
            pdf_path: PDFファイルのパス
            save_index: インデックスを保存するかどうか
            index_name: 保存するインデックスの名前
        """
        # インデックス名のデフォルト値を設定
        if index_name is None:
            basename = os.path.basename(pdf_path)
            index_name = os.path.splitext(basename)[0]

        # PDFからDocumentオブジェクトを作成
        documents = self.document_processor.load_documents(pdf_path)
        
        print(f"🧠 LlamaIndexでインデックスを作成しています...")
        
        # VectorStoreIndexを作成
        self.index = VectorStoreIndex.from_documents(documents)
        
        # 日本語回答用のカスタムプロンプト（より強力）
        qa_prompt_tmpl = (
            "あなたは日本語で回答するアシスタントです。\n"
            "以下のコンテキスト情報を参照して、質問に日本語で答えてください。\n"
            "\n"
            "コンテキスト情報:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "\n"
            "重要な指示:\n"
            "1. 必ず日本語で回答してください\n"
            "2. コンテキスト情報に基づいて回答してください\n"
            "3. 情報がない場合は「提供された情報からは回答できません」と日本語で答えてください\n"
            "4. 英語で回答することは絶対に避けてください\n"
            "\n"
            "質問: {query_str}\n"
            "回答（日本語）: "
        )
        qa_prompt = PromptTemplate(qa_prompt_tmpl)
        
        # QueryEngineを作成（日本語プロンプト使用）
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,  # top_k設定
            response_mode="compact",
            text_qa_template=qa_prompt
        )
        
        self.is_index_built = True
        print(f"  ✅ インデックスの作成が完了しました")

        # インデックスを保存
        if save_index:
            storage_dir = os.path.join(self.models_dir, f"{index_name}_llamaindex")
            os.makedirs(storage_dir, exist_ok=True)
            
            self.index.storage_context.persist(persist_dir=storage_dir)
            print(f"💾 インデックスを保存しました: {storage_dir}")

    def load_index(self, index_name: str) -> None:
        """
        保存されたインデックスを読み込む
        
        Args:
            index_name: インデックスの名前
        """
        storage_dir = os.path.join(self.models_dir, f"{index_name}_llamaindex")
        
        if not os.path.exists(storage_dir):
            raise FileNotFoundError(f"インデックスディレクトリが見つかりません: {storage_dir}")

        print(f"📂 インデックスを読み込んでいます: {storage_dir}")
        
        # StorageContextから読み込み
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        
        # インデックスを読み込み
        self.index = load_index_from_storage(storage_context)
        
        # 日本語回答用のカスタムプロンプト（より強力）
        qa_prompt_tmpl = (
            "あなたは日本語で回答するアシスタントです。\n"
            "以下のコンテキスト情報を参照して、質問に日本語で答えてください。\n"
            "\n"
            "コンテキスト情報:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "\n"
            "重要な指示:\n"
            "1. 必ず日本語で回答してください\n"
            "2. コンテキスト情報に基づいて回答してください\n"
            "3. 情報がない場合は「提供された情報からは回答できません」と日本語で答えてください\n"
            "4. 英語で回答することは絶対に避けてください\n"
            "\n"
            "質問: {query_str}\n"
            "回答（日本語）: "
        )
        qa_prompt = PromptTemplate(qa_prompt_tmpl)
        
        # QueryEngineを作成（日本語プロンプト使用）
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
            text_qa_template=qa_prompt
        )
        
        self.is_index_built = True
        print(f"  ✅ インデックスの読み込みが完了しました")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        質問に対して回答を生成する
        
        Args:
            question: 質問テキスト
            
        Returns:
            回答とその生成に使用された情報を含むディクショナリ
        """
        if not self.is_index_built or self.query_engine is None:
            raise ValueError("インデックスが構築されていません。先にbuild_index_from_pdfまたはload_indexを呼び出してください。")
        
        # QueryEngineで質問に回答
        response = self.query_engine.query(question)
        
        # ソース情報を抽出
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for i, node in enumerate(response.source_nodes):
                source_info = {
                    "document": {
                        "id": node.node.node_id,
                        "content": node.node.text,
                        "metadata": node.node.metadata if node.node.metadata else {}
                    },
                    "score": getattr(node, 'score', 0.0)
                }
                sources.append(source_info)
        
        return {
            "answer": str(response),
            "sources": sources,
            "context": self._format_context(sources),
        }

    def _format_context(self, sources: List[Dict[str, Any]]) -> str:
        """
        ソース情報をコンテキスト文字列に整形する
        
        Args:
            sources: ソース情報のリスト
            
        Returns:
            整形されたコンテキスト文字列
        """
        contexts = []
        for i, source in enumerate(sources):
            doc = source["document"]
            score = source["score"]
            content = doc["content"]
            metadata = doc["metadata"]
            
            source_name = metadata.get("source", "unknown")
            page = metadata.get("page", 0)
            
            context = f"[ドキュメント {i+1}] (ソース: {source_name}, ページ: {page}, スコア: {score:.4f})\\n{content}\\n"
            contexts.append(context)
        
        return "\\n".join(contexts)
