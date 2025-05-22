#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text RAG Module
===============
LlamaIndexを使用したテキストファイル用RAGシステム

主要なクラス:
- RAGConfig: システム設定管理
- TextRAGWorkflow: ワークフロー全体の管理
- TextRAGPipeline: RAGパイプライン実装
- WorkflowLogger: ログ管理
"""

from .config import RAGConfig
from .workflow import TextRAGWorkflow
from .pipeline import TextRAGPipeline
from .logger import WorkflowLogger
from .display import ResultDisplayer
from .handlers import QuestionAnswerHandler
from .managers import IndexManager
from .processors import TextFileProcessor
from .pdf_processor import PDFFileProcessor
from .document_factory import DocumentProcessorFactory

# init.pyはモジュールのnamespaceを定義するためのファイル
__all__ = [
    "RAGConfig",
    "TextRAGWorkflow",
    "TextRAGPipeline",
    "WorkflowLogger",
    "ResultDisplayer",
    "QuestionAnswerHandler",
    "IndexManager",
    "TextFileProcessor",
    "PDFFileProcessor",
    "DocumentProcessorFactory",
]

__version__ = "1.0.0"
