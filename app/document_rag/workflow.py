#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGワークフロー管理
==================
テキストRAGワークフロー全体の実行を管理するメインモジュール
"""

import os

from .config import RAGConfig
from .logger import WorkflowLogger
from .display import ResultDisplayer
from .handlers import QuestionAnswerHandler
from .managers import IndexManager
from .pipeline import TextRAGPipeline


class TextRAGWorkflow:
    """テキストRAGワークフロー全体を管理するメインクラス"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = WorkflowLogger()
        self.displayer = ResultDisplayer(self.logger)
        self.qa_handler = QuestionAnswerHandler(config, self.logger, self.displayer)

    def run(self):
        """RAGワークフロー全体を実行"""
        file_type = "テキスト" if self.config.is_text else "PDF" if self.config.is_pdf else "ドキュメント"
        print(f"🚀 {file_type}ファイル用 LlamaIndex RAGシステムを開始します")
        print(f"📄 対象ファイル: {self.config.document_path}")
        print(f"📝 ファイルタイプ: {file_type} ({self.config.file_extension})")
        print(f"🤖 使用LLMモデル: {self.config.llm_model}")

        # ステップ1: ファイル存在チェック
        if not self._validate_input_file():
            return

        # ステップ2: RAGパイプラインの初期化
        pipeline = TextRAGPipeline(self.config, self.logger)

        # ステップ3: インデックス準備
        index_manager = IndexManager(self.config, self.logger)
        if not index_manager.prepare_index(pipeline):
            self.logger.log_error("インデックスの準備に失敗しました")
            return

        # ステップ4: システム準備完了
        self.logger.log_stage(
            "システム準備完了", "RAGシステムが質問を受け付ける準備ができました"
        )
        file_type = "テキスト" if self.config.is_text else "PDF" if self.config.is_pdf else "ドキュメント"
        self.logger.log_success(f"{file_type}ファイル用 RAGシステムの準備が完了しました！")

        # ステップ5: 質問応答実行
        if self.config.interactive:
            self.qa_handler.run_interactive_mode(pipeline)
        else:
            self.qa_handler.run_sample_mode(pipeline)

    def _validate_input_file(self) -> bool:
        """入力ファイルの存在と形式をチェック"""
        if not os.path.exists(self.config.document_path):
            self.logger.log_error(
                f"指定されたファイルが見つかりません: {self.config.document_path}"
            )
            return False
        
        # サポートされているファイル形式かチェック
        from .document_factory import DocumentProcessorFactory
        if not DocumentProcessorFactory.is_supported_file(self.config.document_path):
            supported_exts = ", ".join(DocumentProcessorFactory.get_supported_extensions())
            self.logger.log_error(
                f"サポートされていないファイル形式です: {self.config.file_extension}\n"
                f"サポートされている形式: {supported_exts}"
            )
            return False
            
        return True
