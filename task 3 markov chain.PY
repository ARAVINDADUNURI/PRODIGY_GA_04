import random
def build_markov_chain(text, n=1):
    """Builds a Markov Chain dictionary from given text."""
    Words = text.split()
    chain = {}
    for i in range(len(Words)-n):
        key = tuple(Words[i:i+n])
        next_word = Words[i+n]
        chain.setdefault(key,[]).append(next_word)
        return chain
def generate_text(chain, length=50):
    """Generates text of a given length using the Markov chain."""
    key = random.choice(list(chain.keys()))
    result = list(key)

    for _ in range(length - len(key)):
        next_words = chain.get(key)
        if not next_words:
            break
        next_word = random.choice(next_words)
        result.append(next_word)
        key = tuple(result[-len(key):])
        return ' '.join(result)

#Example

if __name__ == "__main__":
    sample_text = """Artificial intelligence is transforming the world. Machine learning is a branch of artificial intelligence.
    Markov chains can generate random but realistic sequences."""
    chain = build_markov_chain(sample_text, n=2)
    generated = generate_text(chain,length=20)
    print(generated)