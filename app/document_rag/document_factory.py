#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ドキュメントプロセッサーファクトリー
=================================
ファイル形式に応じて適切なプロセッサーを提供するファクトリークラス
"""

import os
from typing import List, Union

from llama_index.core import Document

from .logger import WorkflowLogger
from .processors import TextFileProcessor
from .pdf_processor import PDFFileProcessor


class DocumentProcessorFactory:
    """ファイル形式に応じて適切なプロセッサーを提供するファクトリークラス"""

    def __init__(self, logger: WorkflowLogger):
        self.logger = logger
        self.text_processor = TextFileProcessor(logger)
        self.pdf_processor = PDFFileProcessor(logger)

    def get_processor_for_file(self, file_path: str):
        """
        ファイル拡張子に基づいて適切なプロセッサーを返す

        Args:
            file_path: ファイルのパス

        Returns:
            適切なプロセッサーインスタンス
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".txt":
            self.logger.log_info("テキストファイルプロセッサーを選択")
            return self.text_processor
        elif file_ext == ".pdf":
            self.logger.log_info("PDFファイルプロセッサーを選択")
            return self.pdf_processor
        else:
            raise ValueError(f"サポートされていないファイル形式です: {file_ext}")

    def load_document(self, file_path: str) -> List[Document]:
        """
        ファイル形式を自動判別してドキュメントを読み込む

        Args:
            file_path: ファイルのパス

        Returns:
            LlamaIndexのDocumentオブジェクトのリスト
        """
        self.logger.log_stage(
            "ステージ2: ドキュメント読み込み", 
            f"ファイル形式を判別して適切な方法で読み込み"
        )

        processor = self.get_processor_for_file(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".txt":
            return processor.load_text_file(file_path)
        elif file_ext == ".pdf":
            return processor.load_pdf_file(file_path)
        else:
            # この分岐は理論上到達しないが、安全のため
            raise ValueError(f"サポートされていないファイル形式です: {file_ext}")

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """サポートされているファイル拡張子のリストを返す"""
        return [".txt", ".pdf"]

    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        """ファイルがサポートされているかどうかをチェック"""
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in DocumentProcessorFactory.get_supported_extensions()
