#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
エンベディングモジュール
====================
テキストデータをベクトル化し、ベクトルDBに格納するためのモジュール
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

class Embedder:
    """テキストをベクトル化してFAISSインデックスに格納するクラス"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        """
        Embedderのコンストラクタ
        
        Args:
            model_name: 埋め込みモデルの名前
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.index = None
        self.documents = []
    
    def create_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        文書リストからベクトルインデックスを作成する
        
        Args:
            documents: ドキュメントのリスト（各ドキュメントはディクショナリ形式）
        """
        self.documents = documents
        texts = [doc["content"] for doc in documents]
        
        print(f"🧠 {self.model_name}モデルでテキストをベクトル化しています...")
        embeddings = self.embeddings.embed_documents(texts)
        
        # NumPy配列に変換
        embeddings_np = np.array(embeddings).astype("float32")
        
        # FAISSインデックスを作成
        dimension = embeddings_np.shape[1]  # 埋め込みの次元数
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        
        print(f"  ✅ {len(documents)}個のドキュメントのベクトル化が完了しました")
        print(f"  ベクトル次元数: {dimension}")
    
    def save_index(self, directory: str, name: str = "faiss_index") -> None:
        """
        FAISSインデックスをファイルに保存する
        
        Args:
            directory: 保存先ディレクトリのパス
            name: インデックスファイルの名前
        """
        if self.index is None:
            raise ValueError("インデックスが作成されていません。save_indexの前にcreate_indexを呼び出してください。")
            
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, f"{name}.index")
        
        faiss.write_index(self.index, index_path)
        print(f"💾 インデックスを保存しました: {index_path}")
    
    def load_index(self, directory: str, name: str = "faiss_index") -> None:
        """
        FAISSインデックスをファイルから読み込む
        
        Args:
            directory: インデックスファイルのディレクトリパス
            name: インデックスファイルの名前
        """
        index_path = os.path.join(directory, f"{name}.index")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"インデックスファイルが見つかりません: {index_path}")
            
        self.index = faiss.read_index(index_path)
        print(f"📂 インデックスを読み込みました: {index_path}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        クエリテキストに類似するドキュメントを検索する
        
        Args:
            query: 検索クエリテキスト
            top_k: 返す結果の数
            
        Returns:
            検索結果のドキュメントリスト（類似度順）
        """
        if self.index is None:
            raise ValueError("インデックスが作成されていません。searchの前にcreate_indexかload_indexを呼び出してください。")
            
        if not self.documents:
            raise ValueError("ドキュメントが読み込まれていません。")
        
        # クエリをベクトル化
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype("float32")
        
        # 検索を実行
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        # 検索結果を構築
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # インデックスの範囲チェック
                doc = self.documents[idx]
                results.append({
                    "document": doc,
                    "score": float(distances[0][i]),
                })
        
        return results