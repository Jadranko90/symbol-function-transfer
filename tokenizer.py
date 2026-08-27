"""Text tokens <-> integer ids. Vocabulary is built from the corpus on disk, so
it always matches exactly the material that will be trained on."""

PAD_TOKEN = "<PAD>"


class Tokenizer:
    def __init__(self, corpus):
        vocab = set()
        for unit in corpus:
            vocab.update(unit)
        tokens = [PAD_TOKEN] + sorted(vocab)
        self.stoi = {tok: i for i, tok in enumerate(tokens)}
        self.itos = {i: tok for i, tok in enumerate(tokens)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.vocab_size = len(tokens)

    def encode(self, unit):
        return [self.stoi[tok] for tok in unit]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def load_corpus_file(path="corpus.txt"):
    """One unit per line, tokens space-separated."""
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


if __name__ == "__main__":
    tok = Tokenizer(load_corpus_file("corpus.txt"))
    print(f"vocab size: {tok.vocab_size}   PAD id: {tok.pad_id}")
