#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGシステムの設定管理
===================
RAGシステムで使用する全ての設定を一元管理するモジュール
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class RAGConfig:
    """RAGシステムの設定を管理するクラス"""

    # ファイル設定
    document_path: str  # テキストファイルまたはPDFファイルのパス

    # モデル設定
    llm_model: str = "gemma:7b"

    # インデックス設定
    rebuild_index: bool = False
    save_index: bool = True
    similarity_top_k: int = 3
    response_mode: str = "compact"

    # 実行設定
    interactive: bool = True

    # 日本語プロンプトテンプレート
    japanese_prompt_template: str = (
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

    # サンプル質問 非対話モードの場合はこの質問が実行される
    sample_questions: List[str] = field(
        default_factory=lambda: [
            "このドキュメントの内容を教えて",
            "この文書の主なトピックは何ですか？",
            "重要なポイントを3つ教えて",
            "このドキュメントで説明されている概念は何ですか？",
        ]
    )

    # 後方互換性のためのプロパティ
    @property
    def text_path(self) -> str:
        """後方互換性のためのテキストパスプロパティ"""
        return self.document_path

    @property
    def index_name(self) -> str:
        """インデックス名を取得"""
        return os.path.splitext(os.path.basename(self.document_path))[0]

    @property
    def index_path(self) -> str:
        """インデックスパスを取得"""
        return os.path.join("models", f"{self.index_name}_llamaindex")
    
    @property
    def file_extension(self) -> str:
        """ファイル拡張子を取得"""
        return os.path.splitext(self.document_path)[1].lower()
    
    @property
    def is_pdf(self) -> bool:
        """PDFファイルかどうかを判定"""
        return self.file_extension == ".pdf"
    
    @property
    def is_text(self) -> bool:
        """テキストファイルかどうかを判定"""
        return self.file_extension == ".txt"
