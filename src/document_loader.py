#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ドキュメントローダーモジュール
=========================
PDFファイルからテキストを抽出し、チャンクに分割するためのモジュール
"""

import os
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class DocumentProcessor:
    """PDFファイルからテキストを抽出し、チャンクに分割するクラス"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        DocumentProcessorのコンストラクタ

        Args:
            chunk_size: チャンクサイズ（文字数）
            chunk_overlap: チャンク間のオーバーラップ（文字数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_and_split(self, file_path: str) -> List[Dict[str, Any]]:
        """
        PDFファイルを読み込み、テキストをチャンクに分割する
        処理の流れ
        1. バリデーションを行う（ファイルが存在するか、PDFファイルかどうか）
        2. PDFファイルを読み込む
        3. テキストをチャンクに分割する
        4. チャンクをディクショナリに変換して返す

        Args:
            file_path: PDFファイルのパス

        Returns:
            チャンク分割されたドキュメントのリスト
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"サポートされていないファイル形式です: {file_path}")

        print(f"📄 PDFファイルを読み込んでいます: {file_path}")

        # PDFファイルからテキストを抽出
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        print(f"  抽出されたページ数: {len(documents)}")
        print(
            f"  読み込まれたドキュメントの最初の1件（最大500文字）: {str(documents[0])[:500] if documents else 'None'}"
        )

        # テキストをチャンクに分割
        chunks = self.text_splitter.split_documents(documents)

        print(f"  生成されたチャンク数: {len(chunks)}")
        if chunks:
            print(
                f"  最初のチャンクの内容（最大500文字）: {str(chunks[0].page_content)[:500]}"
            )
            print(f"  最初のチャンクのメタデータ: {chunks[0].metadata}")

        # チャンクをディクショナリに変換して返す
        result = []
        for i, chunk in enumerate(chunks):
            if i < 2:  # 最初の2チャンクについて情報出力
                print(f"  処理中のチャンク {i}:")
                print(f"    content（最大200文字）: {chunk.page_content[:200]}")
                print(f"    metadata: {chunk.metadata}")
            result.append(
                {
                    "id": f"chunk_{i}",
                    "content": chunk.page_content,
                    "metadata": {
                        "source": chunk.metadata.get("source", "unknown"),
                        "page": chunk.metadata.get("page", 0),
                    },
                }
            )

        return result
