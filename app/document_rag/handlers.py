#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
質問応答処理
===========
対話モードとサンプルモードでの質問応答処理を管理するモジュール
"""

from typing import TYPE_CHECKING

from .config import RAGConfig
from .logger import WorkflowLogger
from .display import ResultDisplayer

if TYPE_CHECKING:
    from .pipeline import TextRAGPipeline


class QuestionAnswerHandler:
    """質問応答処理を管理するクラス"""

    def __init__(
        self, config: RAGConfig, logger: WorkflowLogger, displayer: ResultDisplayer
    ):
        self.config = config
        self.logger = logger
        self.displayer = displayer

    def run_interactive_mode(self, pipeline: "TextRAGPipeline"):
        """対話モードで質問応答を実行"""
        print("\n" + "=" * 60)
        print("🔍 テキストファイルに関する質問ができます。")
        print("終了するには 'exit' または 'quit' と入力してください。")
        print("=" * 60)

        while True:
            print("\n" + "-" * 40)
            question = input("質問を入力してください: ")

            if question.lower() in ["exit", "quit", "終了"]:
                print("\n👋 ありがとうございました！")
                break

            if not question.strip():
                continue

            self._process_question(pipeline, question)
            print("\n" + "=" * 60)

    def run_sample_mode(self, pipeline: "TextRAGPipeline"):
        """サンプル質問で実行"""
        for question in self.config.sample_questions:
            print(f"\n質問: {question}")
            print("-" * 40)
            self._process_question(pipeline, question)
            print("\n" + "=" * 60)

    def _process_question(self, pipeline: "TextRAGPipeline", question: str):
        """個別の質問を処理"""
        try:
            result = pipeline.answer_question(question)
            self.displayer.display_result(result)
        except Exception as e:
            self.logger.log_error(f"質問処理中にエラーが発生しました: {str(e)}")
