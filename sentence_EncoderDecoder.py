class EncoderDecoder(object):
    def __init__(self):
        # word_to_idの辞書
        self.w2i = {}
        # id_to_wordの辞書
        self.i2w = {}
        # 予約語(パディング, 文章の始まり)
        self.special_chars = ['<pad>', '<s>', '</s>', '<unk>']
        self.bos_char = self.special_chars[1]
        self.eos_char = self.special_chars[2]
        self.oov_char = self.special_chars[3]

    # コールされる関数
    def __call__(self, sentence):
        return self.transform(sentence)

    # 辞書作成
    def fit(self, sentences):
        self._words = set()

        # 未知の単語の集合を作成する
        for sentence in sentences:
            self._words.update(sentence)

        # 予約語分ずらしてidを振る
        self.w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._words)}

        # 予約語を辞書に追加する(<pad>:0, <s>:1, </s>:2, <unk>:3)
        for i, w in enumerate(self.special_chars):
            self.w2i[w] = i

        # word_to_idの辞書を用いてid_to_wordの辞書を作成する
        self.i2w = {i: w for w, i in self.w2i.items()}

    # 読み込んだデータをまとめてidに変換する
    def transform(self, sentences, bos=False, eos=False):
        output = []
        # 指定があれば始まりと終わりの記号を追加する
        for sentence in sentences:
            if bos:
                sentence = [self.bos_char] + sentence
            if eos:
                sentence = sentence + [self.eos_char]
            output.append(self.encode(sentence))

        return output

    # 1文ずつidにする
    def encode(self, sentence):
        output = []
        for w in sentence:
            if w not in self.w2i:
                idx = self.w2i[self.oov_char]
            else:
                idx = self.w2i[w]
            output.append(idx)

        return output

    # １文ずつ単語リストに直す
    def decode(self, sentence):
        return [self.i2w[id] for id in sentence]
    
    if __name__ == '__main__':
        import MeCab
        import pandas as pd
        m = MeCab.Tagger('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
        df = pd.read_csv("data/lyrics.csv")
        for text in df['lyrics']:
            text = text.replace('　','')
            print(text)
            texts = text.split("/")
            for i in texts:
                if i == "":
                    continue
                result = m.parse(i).strip().split(" ")
                print(result)
        
