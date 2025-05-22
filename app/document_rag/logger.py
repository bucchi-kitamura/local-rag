#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ワークフローログ管理
==================
RAGワークフローの進行状況とメッセージを管理するモジュール
"""


class WorkflowLogger:
    """ワークフローのログ出力を管理するクラス"""

    @staticmethod
    def log_stage(stage_name: str, description: str = ""):
        """RAGワークフローの段階をログ出力する"""
        print(f"\n{'=' * 60}")
        print(f"🔄 RAGワークフロー: {stage_name}")
        if description:
            print(f"   {description}")
        print(f"{'=' * 60}")

    @staticmethod
    def log_substage(substage_name: str, description: str = ""):
        """RAGワークフローのサブ段階をログ出力する"""
        print(f"\n{'─' * 40}")
        print(f"⚙️  {substage_name}")
        if description:
            print(f"   {description}")
        print(f"{'─' * 40}")

    @staticmethod
    def log_success(message: str, processing_time: float = None):
        """成功メッセージをログ出力"""
        if processing_time is not None:
            print(f"  ✅ {message} (処理時間: {processing_time:.2f}秒)")
        else:
            print(f"  ✅ {message}")

    @staticmethod
    def log_info(message: str):
        """情報メッセージをログ出力"""
        print(f"  📄 {message}")

    @staticmethod
    def log_error(message: str):
        """エラーメッセージをログ出力"""
        print(f"❌ エラー: {message}")

    @staticmethod
    def log_warning(message: str):
        """警告メッセージをログ出力"""
        print(f"⚠️ {message}")
