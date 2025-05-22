#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ドキュメントファイル用 LlamaIndex RAG実行スクリプト（モジュール版）
========================================================
モジュール化されたdocument_ragパッケージを使用してRAGシステムを実行

ワークフロー:
1. システム初期化 (LLMモデルと埋め込みモデルの設定)
2. インデックス準備 (構築または読み込み)
3. 質問応答処理 (対話モードまたはサンプル実行)
"""

import argparse
from pathlib import Path
from typing import List, Optional

from document_rag import RAGConfig, TextRAGWorkflow


def get_data_directory() -> Path:
    """データディレクトリのパスを取得"""
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    return workspace_root / "data"


def list_available_documents(data_dir: Path) -> List[Path]:
    """利用可能なドキュメントファイルをリストアップ"""
    supported_extensions = {".txt", ".pdf"}
    documents = []

    if not data_dir.exists():
        print(f"データディレクトリが見つかりません: {data_dir}")
        return documents

    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            documents.append(file_path)

    return sorted(documents)


def select_document(documents: List[Path]) -> Optional[Path]:
    """ユーザーにドキュメントを選択させる"""
    if not documents:
        print("利用可能なドキュメントファイルがありません。")
        return None

    print("\n利用可能なドキュメントファイル:")
    for i, doc in enumerate(documents, 1):
        file_size = doc.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size}B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f}KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f}MB"

        print(f"  {i}. {doc.name} ({size_str})")

    while True:
        try:
            choice = input(
                f"\nファイルを選択してください (1-{len(documents)}): "
            ).strip()
            if not choice:
                continue

            index = int(choice) - 1
            if 0 <= index < len(documents):
                return documents[index]
            else:
                print(f"1から{len(documents)}の間の数字を入力してください。")
        except ValueError:
            print("有効な数字を入力してください。")
        except KeyboardInterrupt:
            print("\n処理を中断しました。")
            return None


def main(
    document_path: Optional[str] = None,
    llm_model: str = "gemma:7b",
    rebuild_index: bool = False,
    interactive: bool = True,
) -> None:
    """
    ドキュメントファイル（テキストまたはPDF）に対するLlamaIndex RAG質問応答システムを実行する

    Args:
        document_path: ドキュメントファイルのパス（Noneの場合は選択画面を表示）
        llm_model: 使用するLLMモデル名
        rebuild_index: インデックスを再構築するかどうか
        interactive: 対話モードで実行するかどうか
    """
    # ドキュメントパスが指定されていない場合は選択画面を表示
    if document_path is None:
        data_dir = get_data_directory()
        available_docs = list_available_documents(data_dir)
        selected_doc = select_document(available_docs)

        if selected_doc is None:
            print("ドキュメントが選択されませんでした。処理を終了します。")
            return

        document_path = str(selected_doc)

    # 設定を作成
    config = RAGConfig(
        document_path=document_path,
        llm_model=llm_model,
        rebuild_index=rebuild_index,
        interactive=interactive,
    )

    # ワークフローを実行
    workflow = TextRAGWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    # ステップ1: コマンドライン引数パーサーの初期化
    parser = argparse.ArgumentParser(
        description="LlamaIndexを使用したドキュメントファイル（テキスト/PDF）に対するRAG質問応答システム"
    )

    # ステップ2: オプション引数の定義（document_pathを必須から任意に変更）
    parser.add_argument(
        "document_path",
        type=str,
        nargs="?",  # 任意の位置引数
        help="ドキュメントファイルのパス（.txt または .pdf）。指定しない場合は選択画面を表示",
    )

    # ステップ3: その他のオプション引数の定義
    parser.add_argument(
        "--llm_model", type=str, default="gemma:7b", help="使用するLLMモデル名"
    )
    parser.add_argument(
        "--rebuild_index", action="store_true", help="インデックスを再構築する"
    )
    parser.add_argument(
        "--no_interactive", action="store_true", help="対話モードで実行しない"
    )

    # ステップ4: コマンドライン引数の解析
    args = parser.parse_args()

    # ステップ5: 解析された引数を使ってmain関数を実行
    # 注意: interactive は no_interactive の論理反転
    main(
        document_path=args.document_path,
        llm_model=args.llm_model,
        rebuild_index=args.rebuild_index,
        interactive=not args.no_interactive,  # デフォルトで対話モード有効
    )
