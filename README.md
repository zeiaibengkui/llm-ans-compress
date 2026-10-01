# LLM-ans-compress

Compress text with arithmetic coding where the symbol probabilities come either
from the text itself (static model) or from a causal language model (LLM mode).

Output files are gzip containers: 4-byte length + JSON metadata + payload.

## Usage

```bash
uv sync                            # numpy, torch, transformers

uv run python src/ans.py           # static model -> data/encoded.gz
uv run python src/ans.py --llm     # LLM model    -> data/encoded_llm.gz
```

Run from the repo root; both read `data/This is also a kind of life.md`.

As a library:

```python
from ans import compress, decompress

blob = compress(text, method="llm", model="Qwen/Qwen3-0.6B", topk=256)
assert decompress(blob) == text
```

## Sample result

`data/This is also a kind of life.md` — 2816 chars, 8255 utf-8 bytes, 2041 tokens:

| method | raw payload | output  | notes                                  |
|--------|-------------|---------|----------------------------------------|
| static | 2876 B      | 5538 B  | frequency table stored in metadata     |
| llm    | 1848 B      | 1975 B  | model name only, 7.24 bits/token       |
| xz -6  | —           | 4308 B  | reference baseline (gzip -9: 4439 B)   |

On CPU the LLM mode runs at roughly 0.11–0.15 s/token in each direction
(~4 min encode + ~5 min decode for the sample above).

## Notes

- Encode and decode must use the same device and dtype — the probability streams
  have to match exactly. `llm.py` picks fp32 on CPU, fp16 on XPU/CUDA.
- Intel XPU: install the XPU torch build (plain cp314 wheels stop at
  `torch==2.12.1+xpu` from `https://download.pytorch.org/whl/xpu`; the index must
  not be marked `explicit` in `pyproject.toml`).

## Layout

- `src/ans.py` — 32-bit range coder (`ArithmeticEncoder` / `ArithmeticDecoder`)
  and two probability models:
  - **static** — symbol frequencies counted from the text; the frequency table
    is stored in the file metadata.
  - **llm** — every step asks `Qwen/Qwen3-0.6B` for the next-token distribution,
    quantizes the top-256 tokens into integer frequencies, and codes anything
    outside the top-k through an escape symbol (uniform raw token id). Only the
    model name goes into the metadata — the decoder rebuilds the probabilities
    by running the same model over the decoded prefix.
- `src/llm.py` — `InteractiveGenerator`: step-by-step generation with a KV cache
  where the caller picks each next token; used by both encoder and decoder.
- `src/utils.py` — device detection (XPU > CUDA > MPS > CPU).
