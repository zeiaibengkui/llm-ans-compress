import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from utils import device


class InteractiveGenerator:
    """
    A step-by-step generator where the *caller* chooses each next token.

    Typical loop:
        gen = InteractiveGenerator()
        gen.reset("The capital of France is")
        while not done:
            ids, probs = gen.next_probs(topk=5)   # inspect candidates
            chosen = my_policy(ids, probs)        # <-- your logic here
            gen.append(chosen)                    # feed it back

    A KV cache is kept across calls, so `append` costs one forward pass of a
    single token instead of re-reading the whole sequence each step.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = device,
        dtype: torch.dtype | None = None,
    ):
        if dtype is None:
            # CPU 上 fp32 比 fp16 快得多；加速器上 fp16 省显存
            dtype = (
                torch.float32
                if torch.device(device).type == "cpu"
                else torch.float16
            )

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )

        self.device = device
        self.model.to(self.device).eval()

        # Running state
        self._input_ids: torch.Tensor | None = None  # shape (1, seq_len)
        self._past = None  # KV cache
        self._logits: torch.Tensor | None = None  # logits for the next position

    # ------------------------------------------------------------------ #
    #  State management
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset(self, prompt: str = "") -> None:
        """Start a fresh generation with the given prompt."""
        if prompt:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        else:
            # BOS-only start
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            ids = torch.tensor([[bos]])
        self._input_ids = ids.to(self.device)
        self._advance(self._input_ids)

    @torch.no_grad()
    def append(self, token: int | str) -> None:
        """
        Append the caller-chosen token to the running sequence.

        Accepts either a token id (int) or a raw string (which will be
        tokenized; multi-token strings are appended token-by-token).
        """
        if isinstance(token, str):
            new_ids = self.tokenizer(token, add_special_tokens=False).input_ids
            if not new_ids:
                return
            for tid in new_ids:
                self.append(tid)
            return

        tok = torch.tensor([[token]], device=self.device)
        self._input_ids = torch.cat([self._input_ids, tok], dim=-1)
        self._advance(tok)

    # ------------------------------------------------------------------ #
    #  Inspection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def next_probs(self, topk: int = 5) -> tuple[list[int], list[float]]:
        """
        Return the top-k next-token ids and their probabilities.

        Returns:
            (token_ids, probabilities) — both length `topk`, sorted by prob desc.
        """
        probs = F.softmax(self._logits.float(), dim=-1)
        topk_probs, topk_ids = torch.topk(probs, topk)
        return topk_ids.cpu().tolist(), topk_probs.cpu().tolist()

    @torch.no_grad()
    def full_distribution(self) -> torch.Tensor:
        """Return the entire next-token probability vector (vocab-sized)."""
        return F.softmax(self._logits.float(), dim=-1)

    def current_text(self) -> str:
        """Decode everything generated so far (including the prompt)."""
        return self.tokenizer.decode(self._input_ids[0], skip_special_tokens=True)

    @property
    def seq_len(self) -> int:
        return int(self._input_ids.shape[-1])

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_output_embeddings().weight.shape[0])

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _advance(self, new_ids: torch.Tensor) -> None:
        """Feed tokens not yet in the cache, keep the resulting logits."""
        outputs = self.model(
            input_ids=new_ids,
            past_key_values=self._past,
            use_cache=True,
        )
        self._past = outputs.past_key_values
        self._logits = outputs.logits[0, -1, :]


# ---------------------------------------------------------------------- #
#  Example: caller-driven greedy loop with manual override
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    gen = InteractiveGenerator()
    gen.reset("The capital of France is")

    for step in range(10):
        ids, probs = gen.next_probs(topk=5)
        print(f"\nStep {step} | context: {gen.current_text()!r}")
        for tid, p in zip(ids, probs):
            print(f"  {tid:>7}  {p:.4f}  {gen.tokenizer.decode([tid])!r}")

        # --- YOUR policy goes here -------------------------------------
        # Greedy: pick the most likely token
        chosen = ids[0]

        # Or force a specific word, e.g. to steer the model:
        # chosen = gen.tokenizer.encode(" Paris", add_special_tokens=False)[0]
        # ---------------------------------------------------------------

        gen.append(chosen)

        if chosen == gen.tokenizer.eos_token_id:
            break

    print("\nFinal text:", gen.current_text())
