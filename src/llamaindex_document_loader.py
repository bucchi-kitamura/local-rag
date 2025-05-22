#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LlamaIndexドキュメントローダーモジュール
===================================
PDFファイルからテキストを抽出し、LlamaIndexのDocumentオブジェクトに変換するためのモジュール
"""

import os
from typing import List
from pathlib import Path

from llama_index.core import Document
from llama_index.readers.file import PDFReader


class LlamaIndexDocumentProcessor:
    """PDFファイルからLlamaIndexのDocumentオブジェクトを作成するクラス"""

    def __init__(self):
        """
        LlamaIndexDocumentProcessorのコンストラクタ
        """
        self.pdf_reader = PDFReader()

    def load_documents(self, file_path: str) -> List[Document]:
        """
        PDFファイルを読み込み、LlamaIndexのDocumentオブジェクトのリストを返す
        
        Args:
            file_path: PDFファイルのパス
            
        Returns:
            LlamaIndexのDocumentオブジェクトのリスト
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"サポートされていないファイル形式です: {file_path}")

        print(f"📄 PDFファイルを読み込んでいます: {file_path}")

        # PDFファイルからDocumentオブジェクトを作成
        documents = self.pdf_reader.load_data(Path(file_path))
        
        print(f"  抽出されたドキュメント数: {len(documents)}")
        
        # メタデータにファイルパスを追加
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['source'] = file_path
            doc.metadata['page'] = i + 1  # ページ番号は1から開始
        
        if documents:
            print(f"  最初のドキュメントの内容（最大500文字）: {documents[0].text[:500]}")
            print(f"  最初のドキュメントのメタデータ: {documents[0].metadata}")
        
        return documents
