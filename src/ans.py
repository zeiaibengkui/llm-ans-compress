import gzip
import json
import struct
from bisect import bisect_right
from collections import Counter

PRECISION = 32
FULL = 1 << PRECISION
HALF = FULL >> 1
QUARTER = HALF >> 1
THREE_QUARTER = QUARTER * 3
MASK = FULL - 1
MAX_TOTAL = 1 << 16  # 频率总和上限：归一化后区间宽度至少 QUARTER(2^30)，
                     # total 远小于它才能保证每个符号分到的区间非空

# ---- LLM 模式参数 ------------------------------------------------------ #
LLM_MODEL = "Qwen/Qwen3-0.6B"
LLM_TOPK = 256          # 每步保留的候选 token 数（写进元信息，解码端必须一致）
LLM_TOTAL = 1 << 16     # LLM 概率量化后的频率总和


class BitWriter:
    """按位写入，内部维护 bit buffer。"""

    def __init__(self):
        self.buf = 0
        self.nbits = 0
        self.out = bytearray()

    def write(self, bit: int):
        self.buf = (self.buf << 1) | (bit & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.out.append(self.buf)
            self.buf = 0
            self.nbits = 0

    def write_bit_plus_pending(self, bit: int, pending: int):
        # 写入 bit，再写入 pending 个相反位
        self.write(bit)
        for _ in range(pending):
            self.write(bit ^ 1)

    def flush(self) -> bytes:
        if self.nbits > 0:
            self.buf <<= 8 - self.nbits
            self.out.append(self.buf)
            self.buf = 0
            self.nbits = 0
        return bytes(self.out)


class BitReader:
    """按位读取，内部维护 bit buffer。"""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.buf = 0
        self.nbits = 0

    def read(self) -> int:
        if self.nbits == 0:
            if self.pos >= len(self.data):
                return 0  # 不足时补 0
            self.buf = self.data[self.pos]
            self.pos += 1
            self.nbits = 8
        self.nbits -= 1
        return (self.buf >> self.nbits) & 1


class ArithmeticEncoder:
    """逐符号区间编码器：encode() 每步给定符号区间，最后 finish() 收尾。"""

    def __init__(self):
        self.writer = BitWriter()
        self.low = 0
        self.high = MASK
        self.pending = 0

    def encode(self, cum_low: int, cum_high: int, total: int):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high // total) - 1
        self.low = self.low + (rng * cum_low // total)

        while True:
            if self.high < HALF:
                self.writer.write_bit_plus_pending(0, self.pending)
                self.pending = 0
            elif self.low >= HALF:
                self.writer.write_bit_plus_pending(1, self.pending)
                self.pending = 0
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break

            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1

    def finish(self) -> bytes:
        # 收尾
        self.pending += 1
        if self.low < QUARTER:
            self.writer.write_bit_plus_pending(0, self.pending)
        else:
            self.writer.write_bit_plus_pending(1, self.pending)
        return self.writer.flush()


class ArithmeticDecoder:
    """逐符号区间解码器：decode() 每步给定累计频率表，返回符号下标。"""

    def __init__(self, data: bytes):
        self.reader = BitReader(data)
        self.low = 0
        self.high = MASK
        self.code = 0

        # 先读 PRECISION 位初始化 code
        for _ in range(PRECISION):
            self.code = ((self.code << 1) & MASK) | self.reader.read()

    def decode(self, cum: list[int], total: int) -> int:
        rng = self.high - self.low + 1

        # 当前码值对应的累计频率值
        value = ((self.code - self.low + 1) * total - 1) // rng

        # 二分查找：找到 idx 使得 cum[idx] <= value < cum[idx+1]
        idx = min(bisect_right(cum, value) - 1, len(cum) - 2)

        self._update(cum[idx], cum[idx + 1], total)
        return idx

    def _update(self, cum_low: int, cum_high: int, total: int):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high // total) - 1
        self.low = self.low + (rng * cum_low // total)

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break

            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
            self.code = ((self.code << 1) & MASK) | self.reader.read()


def build_cumulative(freqs):
    """累计频率表，长度为 len(freqs)+1。"""
    cum = [0]
    for f in freqs:
        cum.append(cum[-1] + f)
    return cum


def build_model(data):
    """
    从数据自动统计出 symbols 和 freqs。
    符号按频率降序排列（同频保持首次出现顺序），可直接传给 encode/decode。
    """
    if not data:
        return [], []

    counts = Counter(data)
    symbols = sorted(counts, key=lambda c: counts[c], reverse=True)
    freqs = [counts[c] for c in symbols]

    total = sum(freqs)
    if total > MAX_TOTAL:
        # 等比例缩小，保证每个符号至少为 1，区间不会被压成空
        scale = total / MAX_TOTAL
        freqs = [max(1, int(f / scale)) for f in freqs]
    return symbols, freqs


def arithmetic_encode(symbols, freqs, data):
    """
    symbols: 符号列表，如 ['A', 'B', 'C']
    freqs:   对应频率，如 [3, 2, 1]
    data:    要编码的字符串/列表
    """
    total = sum(freqs)
    cum = build_cumulative(freqs)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    enc = ArithmeticEncoder()
    for ch in data:
        idx = sym_to_idx[ch]
        enc.encode(cum[idx], cum[idx + 1], total)
    return enc.finish()


def arithmetic_decode(symbols, freqs, encoded: bytes, length: int):
    """解码出 length 个符号。"""
    total = sum(freqs)
    cum = build_cumulative(freqs)

    dec = ArithmeticDecoder(encoded)
    return "".join(symbols[dec.decode(cum, total)] for _ in range(length))


# ---- LLM 模式 ---------------------------------------------------------- #
def _quantize(probs: list[float], total: int) -> tuple[list[int], int]:
    """
    概率 -> 整数频率：每个候选至少 1，剩余质量全部留给转义符号（也至少 1）。
    返回 (freqs, escape_freq)。
    """
    k = len(probs)
    scale = total - (k + 1)
    freqs = [max(1, int(p * scale)) for p in probs]
    # Σint(p·scale) ≤ scale，每个补 1 最多再贡献 1，因此 sum(freqs) ≤ total - 1
    return freqs, total - sum(freqs)


_raw_cums: dict[int, list[int]] = {}


def _uniform_cum(size: int) -> list[int]:
    """均匀分布的累计频率表，用于转义符号的原始 token id 兜底编码。"""
    if size not in _raw_cums:
        _raw_cums[size] = build_cumulative([1] * size)
    return _raw_cums[size]


def _llm_step(gen, topk: int, total: int):
    """取下一步的 (候选 id, 量化频率, 转义频率)。编码/解码必须走同一条路径。"""
    cand, probs = gen.next_probs(topk=topk)
    freqs, escape = _quantize(probs, total)
    return cand, freqs, escape


def llm_encode(
    text: str,
    model: str = LLM_MODEL,
    topk: int = LLM_TOPK,
    device=None,
):
    """
    用因果语言模型逐步给出下一个 token 的概率，再用区间编码器压缩。
    不在 top-k 里的 token 走转义符号，再用均匀分布编码原始 id 兜底。
    返回 (payload, meta)；meta 记录了模型名，解码端据此重建概率分布。
    """
    from llm import InteractiveGenerator  # 延迟导入：静态模式无需 torch

    kwargs = {} if device is None else {"device": device}
    gen = InteractiveGenerator(model_name=model, **kwargs)
    ids = gen.tokenizer(text, add_special_tokens=False).input_ids
    gen.reset("")

    enc = ArithmeticEncoder()
    for tid in ids:
        cand, freqs, escape = _llm_step(gen, topk, LLM_TOTAL)
        cum = build_cumulative(freqs + [escape])
        idx = cand.index(tid) if tid in cand else len(cand)
        enc.encode(cum[idx], cum[idx + 1], LLM_TOTAL)
        if idx == len(cand):
            raw = _uniform_cum(gen.vocab_size)
            enc.encode(raw[tid], raw[tid + 1], gen.vocab_size)
        gen.append(tid)

    meta = {
        "mode": "llm",
        "model": model,
        "topk": topk,
        "total": LLM_TOTAL,
        "vocab": gen.vocab_size,
        "m": len(ids),
    }
    return enc.finish(), meta


def llm_decode(
    payload: bytes,
    meta: dict,
    ids_only: bool = False,
    device=None,
):
    """
    按 meta 里的模型名重建概率分布并解码。
    ids_only=True 时返回 token id 列表（可与原文 id 逐一比对）。
    """
    from llm import InteractiveGenerator

    kwargs = {} if device is None else {"device": device}
    gen = InteractiveGenerator(model_name=meta["model"], **kwargs)
    gen.reset("")

    topk, total, vocab = meta["topk"], meta["total"], meta["vocab"]
    dec = ArithmeticDecoder(payload)
    ids = []
    for _ in range(meta["m"]):
        cand, freqs, escape = _llm_step(gen, topk, total)
        cum = build_cumulative(freqs + [escape])
        idx = dec.decode(cum, total)
        if idx < len(cand):
            tid = cand[idx]
        else:
            raw = _uniform_cum(vocab)
            tid = dec.decode(raw, vocab)
        ids.append(tid)
        gen.append(tid)

    return ids if ids_only else gen.tokenizer.decode(ids)


# ---- 打包 / 解包 ------------------------------------------------------- #
def save(payload: bytes, meta: dict, filename: str | None = None) -> bytes:
    """把编码结果和元信息（mode / model / symbols …）打包成 gzip，可选写文件。"""
    raw = json.dumps(meta).encode("utf-8")
    data = gzip.compress(
        struct.pack("<I", len(raw)) + raw + payload, compresslevel=9, mtime=0
    )
    if filename is not None:
        with open(filename, "wb") as f:
            f.write(data)
    return data


def load(data: bytes) -> tuple[bytes, dict]:
    """还原 save 的内容，返回 (payload, meta)。"""
    raw = gzip.decompress(data)
    (n,) = struct.unpack("<I", raw[:4])
    return raw[4 + n :], json.loads(raw[4 : 4 + n].decode("utf-8"))


def compress(
    data: str,
    method: str = "static",
    model: str = LLM_MODEL,
    topk: int = LLM_TOPK,
    filename: str | None = None,
    device=None,
) -> bytes:
    """把文本压成 gzip blob；method="llm" 时用语言模型作为概率模型。"""
    if method == "llm":
        payload, meta = llm_encode(data, model=model, topk=topk, device=device)
    elif method == "static":
        symbols, freqs = build_model(data)
        payload = arithmetic_encode(symbols, freqs, data)
        meta = {
            "mode": "static",
            "length": len(data),
            "symbols": list(symbols),
            "freqs": list(freqs),
        }
    else:
        raise ValueError(f"unknown method: {method!r}")
    return save(payload, meta, filename)


def decompress(blob: bytes, device=None) -> str:
    payload, meta = load(blob)
    if meta["mode"] == "llm":
        return llm_decode(payload, meta, device=device)
    return arithmetic_decode(meta["symbols"], meta["freqs"], payload, meta["length"])


# ---------------- 测试 ----------------
if __name__ == "__main__":
    import sys
    import time

    try:
        with open("data/This is also a kind of life.md", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = "hello world from python"

    if "--llm" in sys.argv:
        # 语言模型做概率模型：逐 token 编码，模型名写进元信息
        size = len(text.encode("utf-8"))
        t0 = time.time()
        payload, meta = llm_encode(text)
        blob = save(payload, meta, "data/encoded_llm.gz")
        t1 = time.time()

        decoded_ids = llm_decode(*load(blob), ids_only=True)
        t2 = time.time()

        from transformers import AutoTokenizer

        ids = AutoTokenizer.from_pretrained(meta["model"])(
            text, add_special_tokens=False
        ).input_ids
        n = max(len(ids), 1)
        print(
            f"LLM mode | {meta['model']} | {len(ids)} tokens "
            f"({size} utf-8 bytes) -> {len(blob)} gz bytes, {len(payload)} raw"
        )
        print(
            f"  {len(payload) * 8 / n:.2f} bits/token, {len(payload) * 8 / size:.2f} "
            f"bits/byte | encode {t1 - t0:.1f}s, decode {t2 - t1:.1f}s"
        )
        if decoded_ids == ids:
            print(f"  token ids match ({len(ids)} tokens)")
        else:
            i = next(
                (i for i, (a, b) in enumerate(zip(ids, decoded_ids)) if a != b),
                min(len(ids), len(decoded_ids)),
            )
            print(f"  token mismatch at {i}: {ids[i : i + 5]} -> {decoded_ids[i : i + 5]}")
    else:
        blob = compress(text, method="static", filename="data/encoded.gz")
        payload, meta = load(blob)
        decoded = decompress(blob)
        print(
            f"Encoded {len(text)} chars over {len(meta['symbols'])} symbols "
            f"into {len(blob)} gz bytes ({len(payload)} raw)"
        )
        if decoded == text:
            print(f"Decoded {len(decoded)} chars: round trip OK")
        else:
            i = next(
                (i for i, (a, b) in enumerate(zip(text, decoded)) if a != b),
                min(len(text), len(decoded)),
            )
            print(
                f"Decoded {len(decoded)} chars: mismatch at {i}: "
                f"{text[i : i + 20]!r} -> {decoded[i : i + 20]!r}"
            )
