# pip install gensim
from gensim.models import Word2Vec

# サンプルのコーパス（トークン化済みの文）
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["Word2Vec", "is", "a", "powerful", "tool"],
    ["Gensim", "makes", "it", "easy", "to", "use", "Word2Vec"],
    ["Skip-gram", "and", "CBOW", "are", "two", "architectures"],
    ["Words", "are", "represented", "as", "vectors"]
]

# モデルの学習（CBOW: sg=0, Skip-gram: sg=1）
model = Word2Vec(
    sentences,
    vector_size=100,  # ベクトルの次元数
    window=2,         # 文脈ウィンドウサイズ
    min_count=1,      # 出現回数の最小値（1ならすべての単語を学習）
    sg=0              # 0=CBOW, 1=Skip-gram
)

# 単語ベクトルの確認
print(model.wv["Word2Vec"])

# 類似語の確認
print(model.wv.most_similar("Word2Vec"))
