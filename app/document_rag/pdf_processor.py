#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDFファイル処理
==============
PDFファイルの読み込みとLlamaIndexドキュメント変換を行うモジュール
"""

import os
import time
from typing import List

from llama_index.core import Document
from llama_index.readers.file import PDFReader

from .logger import WorkflowLogger


class PDFFileProcessor:
    """PDFファイルをLlamaIndexのDocumentオブジェクトに変換するクラス"""

    def __init__(self, logger: WorkflowLogger):
        self.logger = logger
        self.pdf_reader = PDFReader()

    def load_pdf_file(self, file_path: str) -> List[Document]:
        """
        PDFファイルを読み込み、LlamaIndexのDocumentオブジェクトを作成

        Args:
            file_path: PDFファイルのパス

        Returns:
            LlamaIndexのDocumentオブジェクトのリスト
        """
        self.logger.log_substage(
            "PDFドキュメント読み込み", "PDFファイルをDocumentオブジェクトに変換"
        )

        self._validate_file(file_path)

        self.logger.log_info(f"PDFファイルを読み込んでいます: {file_path}")

        # PDFファイルを読み込み
        start_time = time.time()
        
        try:
            # LlamaIndex PDFReaderを使用してPDFを読み込み
            documents = self.pdf_reader.load_data(file=file_path)
            load_time = time.time() - start_time

            # メタデータを拡張
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "source": file_path,
                    "file_type": "pdf",
                    "file_name": os.path.basename(file_path),
                    "page_number": i + 1,  # ページ番号（1から開始）
                    "total_pages": len(documents)
                })

            total_chars = sum(len(doc.text) for doc in documents)
            self.logger.log_success(
                f"PDF読み込み完了: {len(documents)}ページ, {total_chars}文字", 
                load_time
            )
            self.logger.log_info(f"ファイル名: {os.path.basename(file_path)}")
            self.logger.log_info(f"ページ数: {len(documents)}")

            return documents

        except Exception as e:
            self.logger.log_error(f"PDF読み込みエラー: {str(e)}")
            raise

    def _validate_file(self, file_path: str):
        """ファイルの存在と形式をチェック"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"PDFファイルではありません: {file_path}")
