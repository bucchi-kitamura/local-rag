#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGパイプライン
==============
LlamaIndexを使用したテキストRAGパイプラインの実装
"""

import os
import sys
import time
from typing import List

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llamaindex_rag_pipeline import LlamaIndexRAGPipeline
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate

from .config import RAGConfig
from .logger import WorkflowLogger
from .processors import TextFileProcessor
from .document_factory import DocumentProcessorFactory


class TextRAGPipeline(LlamaIndexRAGPipeline):
    """テキストファイル用のRAGパイプライン"""

    def __init__(self, config: RAGConfig, logger: WorkflowLogger):
        logger.log_stage("ステージ1: システム初期化", "LLMモデルと埋め込みモデルの設定")
        super().__init__(llm_model=config.llm_model)
        self.config = config
        self.logger = logger
        self.document_factory = DocumentProcessorFactory(logger)
        logger.log_success("RAGパイプラインの初期化が完了しました")

    def build_index_from_document(
        self,
        document_path: str,
        save_index: bool = True,
        index_name: str = None,
    ) -> None:
        """
        ドキュメントファイル（テキストまたはPDF）からベクトルインデックスを構築する

        Args:
            document_path: ドキュメントファイルのパス
            save_index: インデックスを保存するかどうか
            index_name: 保存するインデックスの名前
        """
        # インデックス名のデフォルト値を設定
        if index_name is None:
            basename = os.path.basename(document_path)
            index_name = os.path.splitext(basename)[0]

        # ドキュメントファイルからDocumentオブジェクトを作成
        documents = self.document_factory.load_document(document_path)

        # ベクトルインデックスを作成
        self._create_vector_index(documents)

        # クエリエンジンを設定
        self._setup_query_engine()

        # インデックスを保存
        if save_index:
            self._save_index(index_name)

    # 後方互換性のためのメソッド
    def build_index_from_text(
        self,
        text_path: str,
        save_index: bool = True,
        index_name: str = None,
    ) -> None:
        """
        テキストファイルからベクトルインデックスを構築する（後方互換性のため）

        Args:
            text_path: テキストファイルのパス
            save_index: インデックスを保存するかどうか
            index_name: 保存するインデックスの名前
        """
        self.build_index_from_document(text_path, save_index, index_name)

    def _create_vector_index(self, documents: List[Document]):
        """ベクトルインデックスを作成"""
        self.logger.log_substage(
            "ベクトルインデックス作成", "LlamaIndexでドキュメントをベクトル化"
        )
        self.logger.log_info("LlamaIndexでインデックスを作成しています...")

        start_time = time.time()
        self.index = VectorStoreIndex.from_documents(documents)
        index_time = time.time() - start_time

        self.logger.log_success("ベクトルインデックス作成完了", index_time)

    def _setup_query_engine(self):
        """クエリエンジンを設定"""
        self.logger.log_substage(
            "クエリエンジン設定", "日本語回答用プロンプトとQueryEngineの設定"
        )

        qa_prompt = PromptTemplate(self.config.japanese_prompt_template)

        # QueryEngineを作成（日本語プロンプト使用）
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.similarity_top_k,
            response_mode=self.config.response_mode,
            text_qa_template=qa_prompt,
        )

        self.is_index_built = True
        self.logger.log_success("QueryEngine設定完了")

    def _save_index(self, index_name: str):
        """インデックスを保存"""
        self.logger.log_substage(
            "インデックス保存", "作成したインデックスをディスクに保存"
        )
        storage_dir = os.path.join(self.models_dir, f"{index_name}_llamaindex")
        os.makedirs(storage_dir, exist_ok=True)

        start_time = time.time()
        self.index.storage_context.persist(persist_dir=storage_dir)
        save_time = time.time() - start_time

        self.logger.log_info(f"インデックスを保存しました: {storage_dir}")
        self.logger.log_success("保存完了", save_time)

    def answer_question(self, question: str):
        """質問応答処理にログを追加"""
        self.logger.log_stage("ステージ3: 質問応答処理", f"質問: {question}")

        self.logger.log_substage("類似文書検索", "質問に関連する文書をベクトル検索")

        start_time = time.time()
        result = super().answer_question(question)
        total_time = time.time() - start_time

        self.logger.log_success("質問応答完了", total_time)
        self.logger.log_info(f"検索された文書数: {len(result['sources'])}")

        return result
