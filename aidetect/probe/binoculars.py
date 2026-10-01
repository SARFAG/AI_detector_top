"""Binoculars: perplexity normalised by cross-perplexity.

    score = perplexity(text | observer) / cross_perplexity(observer, performer)

The denominator is the whole point. Raw perplexity says "this text is
predictable", which is also true of a non-native speaker writing plainly, and
that is why naive perplexity detectors have catastrophic false-positive rates on
those writers. Cross-perplexity measures how surprising the *observer* finds the
*performer's* predictions, which normalises away baseline predictability and
leaves only the part attributable to machine generation.

LOW score => machine-generated.

Reference threshold ~0.901 for the Falcon-7B pair at a low-FPR operating point.
Thresholds are model-pair specific; re-derive yours on held-out data from your
own domain rather than trusting this constant.
"""

from __future__ import annotations

from typing import Optional

OBSERVER_DEFAULT = "tiiuae/falcon-7b"
PERFORMER_DEFAULT = "tiiuae/falcon-7b-instruct"

# Operating point from the original work, tuned for low false-positive rate.
THRESHOLD_LOW_FPR = 0.9015310749276843
THRESHOLD_BALANCED = 0.8536432310785527


class BinocularsUnavailable(RuntimeError):
    """Raised when torch/transformers or the weights are not installed."""


class Binoculars:
    """Lazy wrapper. Nothing is loaded until the first `score()` call."""

    def __init__(
        self,
        observer: str = OBSERVER_DEFAULT,
        performer: str = PERFORMER_DEFAULT,
        device: Optional[str] = None,
        threshold: float = THRESHOLD_LOW_FPR,
        max_length: int = 512,
    ) -> None:
        self.observer_name = observer
        self.performer_name = performer
        self.device = device
        self.threshold = threshold
        self.max_length = max_length
        self._loaded = False
        self._observer = None
        self._performer = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BinocularsUnavailable(
                "Binoculars needs torch and transformers:\n"
                "    pip install torch transformers"
            ) from exc

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._observer = AutoModelForCausalLM.from_pretrained(
            self.observer_name, torch_dtype=dtype).to(self.device).eval()
        self._performer = AutoModelForCausalLM.from_pretrained(
            self.performer_name, torch_dtype=dtype).to(self.device).eval()
        self._loaded = True

    def score(self, text: str) -> float:
        """Return the Binoculars score. LOW means machine-generated."""
        self._load()
        import torch

        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length).to(self.device)
        input_ids = enc.input_ids

        if input_ids.shape[1] < 2:
            raise ValueError("text too short to score")

        with torch.no_grad():
            obs_logits = self._observer(**enc).logits
            prf_logits = self._performer(**enc).logits

        # Shift so position i predicts token i+1.
        obs = obs_logits[:, :-1, :]
        prf = prf_logits[:, :-1, :]
        targets = input_ids[:, 1:]

        # Perplexity of the text under the observer.
        ce = torch.nn.functional.cross_entropy(
            obs.reshape(-1, obs.shape[-1]), targets.reshape(-1), reduction="mean")

        # Cross-perplexity: how surprised the observer is by the performer's
        # full predicted distribution at each position.
        prf_probs = torch.softmax(prf, dim=-1)
        obs_logprobs = torch.log_softmax(obs, dim=-1)
        xce = -(prf_probs * obs_logprobs).sum(dim=-1).mean()

        return float(ce / xce)

    def is_machine(self, text: str) -> bool:
        return self.score(text) < self.threshold

    def as_signal(self, text: str):
        """Return a `Signal` so this slots into the normal ensemble report."""
        from ..signals import Signal

        s = self.score(text)
        # Map distance from threshold onto bounded log-odds. The scale (12.0)
        # is a rough prior; re-derive it on your own held-out data.
        logodds = max(-3.0, min(3.0, (self.threshold - s) * 12.0))
        return Signal(
            name="probe.binoculars",
            layer="probe",
            logodds=logodds,
            evidence=[f"Binoculars={s:.4f} (threshold {self.threshold:.4f}; "
                      f"lower = more machine-like)"],
            detail=s,
        )
