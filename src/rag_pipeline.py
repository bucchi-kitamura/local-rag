#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGパイプラインモジュール
=====================
PDF読み込み、ベクトル化、検索、LLMによる回答生成までのパイプラインを提供
"""

import os
import json
from typing import List, Dict, Any, Optional

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from document_loader import DocumentProcessor
from embedder import Embedder

class RAGPipeline:
    """PDFからRAGパイプラインを構築するクラス"""
    
    def __init__(
        self,
        llm_model: str = "gemma:7b",
        embed_model: str = "intfloat/multilingual-e5-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        models_dir: str = "models",
    ):
        """
        RAGPipelineのコンストラクタ
        
        Args:
            llm_model: 使用するLLMモデル名
            embed_model: 使用する埋め込みモデル名
            chunk_size: チャンクサイズ（文字数）
            chunk_overlap: チャンク間のオーバーラップ（文字数）
            models_dir: モデルディレクトリのパス
        """
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.models_dir = models_dir
        
        # コンポーネントの初期化
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = Embedder(model_name=embed_model)
        self.llm = OllamaLLM(model=llm_model)
        
        # プロンプトテンプレートの設定
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            質問に対して、以下の情報に基づいて回答してください。

            情報:
            {context}

            質問: {question}

            回答:
            """
        )
        
        # LangChain v0.2の新しい記法でチェーンを作成
        self.llm_chain = self.prompt_template | self.llm
        
        # 状態管理用の変数
        self.is_index_built = False
        self.documents = []
        
    def build_index_from_pdf(
        self,
        pdf_path: str,
        save_index: bool = True,
        index_name: Optional[str] = None,
    ) -> None:
        """
        PDFからベクトルインデックスを構築する
        
        Args:
            pdf_path: PDFファイルのパス
            save_index: インデックスを保存するかどうか
            index_name: 保存するインデックスの名前
        """
        # ディレクトリ名を抽出し、インデックス名のデフォルト値として使用
        if index_name is None:
            basename = os.path.basename(pdf_path)
            index_name = os.path.splitext(basename)[0]
        
        # PDFからテキストを抽出してチャンクに分割
        self.documents = self.document_processor.load_and_split(pdf_path)
        
        # ベクトルインデックスを作成
        self.embedder.create_index(self.documents)
        self.is_index_built = True
        
        # インデックスを保存
        if save_index:
            self.embedder.save_index(self.models_dir, name=index_name)
            
            # ドキュメントメタデータも保存
            metadata_path = os.path.join(self.models_dir, f"{index_name}_documents.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            print(f"💾 ドキュメントメタデータを保存しました: {metadata_path}")
    
    def load_index(self, index_name: str) -> None:
        """
        保存されたインデックスを読み込む
        
        Args:
            index_name: インデックスの名前
        """
        # インデックスを読み込む
        self.embedder.load_index(self.models_dir, name=index_name)
        
        # ドキュメントメタデータも読み込む
        metadata_path = os.path.join(self.models_dir, f"{index_name}_documents.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            self.embedder.documents = self.documents
            print(f"📂 ドキュメントメタデータを読み込みました: {metadata_path}")
            self.is_index_built = True
        else:
            raise FileNotFoundError(f"ドキュメントメタデータファイルが見つかりません: {metadata_path}")
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        質問に対して回答を生成する
        
        Args:
            question: 質問テキスト
            top_k: 使用する類似ドキュメント数
            
        Returns:
            回答とその生成に使用された情報を含むディクショナリ
        """
        if not self.is_index_built:
            raise ValueError("インデックスが構築されていません。先にbuild_index_from_pdfまたはload_indexを呼び出してください。")
        
        # 質問に関連するドキュメントを検索
        search_results = self.embedder.search(question, top_k=top_k)
        
        # コンテキスト情報を結合
        contexts = []
        for i, result in enumerate(search_results):
            doc = result["document"]
            score = result["score"]
            content = doc["content"]
            source = doc["metadata"]["source"]
            page = doc["metadata"]["page"]
            
            context = f"[ドキュメント {i+1}] (ソース: {source}, ページ: {page})\n{content}\n"
            contexts.append(context)
        
        context_text = "\n".join(contexts)
        
        # LLMに回答を生成させる
        answer = self.llm_chain.invoke({
            "context": context_text,
            "question": question
        })
        
        return {
            "answer": answer,
            "sources": search_results,
            "context": context_text,
        }