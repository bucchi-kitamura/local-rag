#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
インデックス管理
===============
ベクトルインデックスの構築・読み込み・保存を管理するモジュール
"""

import os
import time
from typing import TYPE_CHECKING

from .config import RAGConfig
from .logger import WorkflowLogger

if TYPE_CHECKING:
    from .pipeline import TextRAGPipeline


class IndexManager:
    """インデックスの構築・読み込み・保存を管理するクラス"""

    def __init__(self, config: RAGConfig, logger: WorkflowLogger):
        self.config = config
        self.logger = logger

    def prepare_index(self, pipeline: "TextRAGPipeline") -> bool:
        """
        インデックスを準備する（構築または読み込み）

        Returns:
            bool: インデックスの準備が成功したかどうか
        """
        if self.config.rebuild_index or not os.path.exists(self.config.index_path):
            return self._build_index(pipeline)
        else:
            return self._load_existing_index(pipeline)

    def _build_index(self, pipeline: "TextRAGPipeline") -> bool:
        """新しいインデックスを構築"""
        file_type = "テキスト" if self.config.is_text else "PDF" if self.config.is_pdf else "ドキュメント"
        self.logger.log_stage(
            "ステージ2: インデックス構築", f"{file_type}からベクトルインデックスを作成"
        )

        try:
            pipeline.build_index_from_document(
                self.config.document_path,
                save_index=self.config.save_index,
                index_name=self.config.index_name,
            )
            return True
        except Exception as e:
            self.logger.log_error(f"インデックス構築に失敗しました: {str(e)}")
            return False

    def _load_existing_index(self, pipeline: "TextRAGPipeline") -> bool:
        """既存のインデックスを読み込み"""
        self.logger.log_stage(
            "ステージ2: インデックス読み込み", "既存のインデックスを読み込み"
        )
        self.logger.log_info("既存のLlamaIndexインデックスを読み込んでいます")

        try:
            start_time = time.time()
            pipeline.load_index(self.config.index_name)
            load_time = time.time() - start_time
            self.logger.log_success("インデックス読み込み完了", load_time)
            return True
        except Exception as e:
            self.logger.log_warning(f"インデックスの読み込みに失敗しました: {str(e)}")
            self.logger.log_info("新しいインデックスを構築します")
            return self._build_index(pipeline)
