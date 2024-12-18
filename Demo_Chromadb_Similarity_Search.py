from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_chroma import Chroma

# 環境変数 OPENAI_API_KEY に OpenAI API Key を設定（.env ファイルの OPENAI_API_KEY からロード）
_ = load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # 埋め込みモデルの準備

collection_name = "mycollection_200" # 検索対象のChromadb コレクション名（事前に作成済み）

top_k = 4 # 取得する検索結果の数の設定

# LangChain の Chromadb VectorStore のインスタンスを作成
vector_store = Chroma(
    client=chromadb.PersistentClient(path="./chroma_db"),
    collection_name=collection_name,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "ip"} 
)

query="ハグリッドがホグワーツへの入学案内書を持ってきたのはいつですか？それはどのような日でしたか？" # クエリーの定義

results =vector_store.similarity_search_with_score(query, k=top_k) # 類似性検索の実行

print("検索結果:")
for i, (doc, score) in enumerate(results, 1):
    print(f"\n=== 結果 {i} ===")
    print(f"ソース: {doc.metadata['source']}")
    print(f"スコア: {score:.4f}")
    print(f"内容: {doc.page_content}")