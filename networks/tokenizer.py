class ByteTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode('utf-8')]

    def decode(self, tokens, skip_special_tokens=True):
        return bytes([i - 3 for i in tokens if i >= 3]).decode('utf-8', errors='replace')
