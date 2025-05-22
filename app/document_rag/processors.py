#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
テキストファイル処理
==================
テキストファイルの読み込みとLlamaIndexドキュメント変換を行うモジュール
"""

import os
import time
from typing import List

from llama_index.core import Document

from .logger import WorkflowLogger


class TextFileProcessor:
    """テキストファイルをLlamaIndexのDocumentオブジェクトに変換するクラス"""

    def __init__(self, logger: WorkflowLogger):
        self.logger = logger

    def load_text_file(self, file_path: str) -> List[Document]:
        """
        テキストファイルを読み込み、LlamaIndexのDocumentオブジェクトを作成

        Args:
            file_path: テキストファイルのパス

        Returns:
            LlamaIndexのDocumentオブジェクトのリスト
        """
        self.logger.log_substage(
            "ドキュメント読み込み", "テキストファイルをDocumentオブジェクトに変換"
        )

        self._validate_file(file_path)

        self.logger.log_info(f"テキストファイルを読み込んでいます: {file_path}")

        # テキストファイルを読み込み
        start_time = time.time()
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        load_time = time.time() - start_time

        # LlamaIndexのDocumentオブジェクトを作成
        document = Document(
            text=content,
            metadata={
                "source": file_path,
                "file_type": "txt",
                "file_name": os.path.basename(file_path),
            },
        )

        self.logger.log_success(f"読み込み完了: {len(content)}文字", load_time)
        self.logger.log_info(f"ファイル名: {os.path.basename(file_path)}")

        return [document]

    def _validate_file(self, file_path: str):
        """ファイルの存在と形式をチェック"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

        if not file_path.lower().endswith(".txt"):
            raise ValueError(f"サポートされていないファイル形式です: {file_path}")
