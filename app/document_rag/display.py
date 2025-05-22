#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
結果表示管理
===========
質問応答結果の表示とフォーマットを管理するモジュール
"""

from typing import Dict, Any

from .logger import WorkflowLogger


class ResultDisplayer:
    """質問応答結果の表示を管理するクラス"""

    def __init__(self, logger: WorkflowLogger):
        self.logger = logger

    def display_result(self, result: Dict[str, Any]):
        """質問応答結果を表示"""
        print("\n📝 **回答:**")
        print(result["answer"])

        # 引用元を表示
        if result["sources"]:
            print("\n📚 **参照元:**")
            for i, source in enumerate(result["sources"]):
                self._display_source(source, i + 1)
                if i < len(result["sources"]) - 1:
                    print("-" * 40)
        else:
            print("\n📚 **参照元:** なし")

    def _display_source(self, source: Dict[str, Any], source_num: int):
        """個別のソース情報を表示"""
        doc = source["document"]
        score = source["score"]
        metadata = doc["metadata"]
        source_name = metadata.get("source", "unknown")
        file_name = metadata.get("file_name", "unknown")

        print(f"--- ソース {source_num} ---")
        print(f"  📄 ファイル: {file_name}")
        print(f"  📁 パス: {source_name}")
        print(f"  🎯 類似度スコア: {score:.4f}")
