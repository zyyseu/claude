"""
Train Qwen3 4B as a reranker using Jina Rerank v3's listwise training approach.

Jina Rerank v3 uses listwise softmax cross-entropy: for each query, the model
scores all candidate documents independently, then a softmax over scores produces
a probability distribution. The loss minimizes cross-entropy against the
normalized relevance labels — encouraging relevant docs to dominate the softmax.

Usage:
    python train_reranker.py \
        --model_name Qwen/Qwen3-4B \
        --train_data /path/to/train.jsonl \
        --output_dir ./output \
        --epochs 3 \
        --batch_size 4 \
        --lr 1e-5
"""

import os
import json
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryGroup:
    query: str
    documents: List[str]
    labels: List[float]          # relevance scores, higher = more relevant
    query_id: str = ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RerankDataset(Dataset):
    """
    Listwise reranking dataset.

    Expected JSONL formats:

    Standard:
        {"query": "...", "documents": ["d1", "d2"], "labels": [1.0, 0.0]}

    MS MARCO style:
        {"query": "...", "passages": [{"text": "...", "is_selected": 1}, ...]}

    Each line is one query group. All documents in a group are scored together
    for the listwise loss.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        max_docs_per_query: int = 20,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_docs_per_query = max_docs_per_query
        self.groups: List[QueryGroup] = []
        self._load(data_path)
        logger.info(f"Loaded {len(self.groups)} query groups from {data_path}")

    def _load(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                query = item["query"]

                if "passages" in item:
                    docs = [p["text"] for p in item["passages"]]
                    labels = [float(p.get("is_selected", p.get("label", 0)))
                              for p in item["passages"]]
                else:
                    docs = item["documents"]
                    labels = item["labels"]

                if len(docs) < 2:
                    continue

                self.groups.append(QueryGroup(
                    query=query,
                    documents=docs[:self.max_docs_per_query],
                    labels=labels[:self.max_docs_per_query],
                    query_id=item.get("query_id", item.get("qid", "")),
                ))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        all_input_ids, all_attention_masks = [], []

        for doc in group.documents:
            text = f"Query: {group.query}\nDocument: {doc}"
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            all_input_ids.append(enc["input_ids"].squeeze(0))
            all_attention_masks.append(enc["attention_mask"].squeeze(0))

        return {
            "input_ids": torch.stack(all_input_ids),        # (num_docs, L)
            "attention_mask": torch.stack(all_attention_masks),
            "labels": torch.tensor(group.labels, dtype=torch.float32),
            "num_docs": len(group.documents),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad each group to the same number of documents within the batch."""
    max_docs = max(item["num_docs"] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]

    input_ids_list, attention_masks_list, labels_list, doc_masks_list = [], [], [], []

    for item in batch:
        n = item["num_docs"]
        if n < max_docs:
            pad = lambda t, v: torch.cat([
                t, torch.full((max_docs - n, *t.shape[1:]), v, dtype=t.dtype)
            ])
            input_ids_list.append(pad(item["input_ids"], 0))
            attention_masks_list.append(pad(item["attention_mask"], 0))
            labels_list.append(pad(item["labels"], 0))
        else:
            input_ids_list.append(item["input_ids"])
            attention_masks_list.append(item["attention_mask"])
            labels_list.append(item["labels"])

        mask = torch.zeros(max_docs, dtype=torch.bool)
        mask[:n] = True
        doc_masks_list.append(mask)

    return {
        "input_ids": torch.stack(input_ids_list),            # (B, max_docs, L)
        "attention_mask": torch.stack(attention_masks_list),
        "labels": torch.stack(labels_list),                  # (B, max_docs)
        "doc_mask": torch.stack(doc_masks_list),             # (B, max_docs)
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3Reranker(nn.Module):
    """
    Qwen3 backbone + linear scoring head for reranking.

    Uses the last non-padding token's hidden state to produce a scalar
    relevance score for each (query, document) pair.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.score_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        nn.init.normal_(self.score_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch_size, num_docs, seq_len)
            attention_mask: (batch_size, num_docs, seq_len)

        Returns:
            scores: (batch_size, num_docs)
        """
        B, N, L = input_ids.shape

        flat_input_ids = input_ids.view(B * N, L)
        flat_attention_mask = attention_mask.view(B * N, L)

        outputs = self.base_model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        # Last non-padding token per sequence
        seq_lengths = flat_attention_mask.sum(dim=1) - 1           # (B*N,)
        batch_indices = torch.arange(B * N, device=input_ids.device)
        last_hidden = outputs.last_hidden_state[batch_indices, seq_lengths]  # (B*N, H)

        scores = self.score_head(last_hidden).squeeze(-1)          # (B*N,)
        return scores.view(B, N)

    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.base_model.save_pretrained(save_path)
        torch.save(self.score_head.state_dict(), os.path.join(save_path, "score_head.pt"))
        logger.info(f"Model saved to {save_path}")


# ---------------------------------------------------------------------------
# Listwise loss functions
# ---------------------------------------------------------------------------

def listwise_softmax_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Listwise softmax cross-entropy (Jina Rerank v3's core loss).

    For each query group:
      P_model(i)  = softmax(scores / τ)
      P_target(i) = labels⁺ / Σ labels⁺
      loss        = -Σ P_target(i) · log P_model(i)

    Args:
        scores:      (B, N)  predicted scores
        labels:      (B, N)  relevance labels (non-negative)
        mask:        (B, N)  True for real docs, False for padding
        temperature: softmax temperature
    """
    scores = scores / temperature
    scores = scores.masked_fill(~mask, float("-inf"))

    log_probs = F.log_softmax(scores, dim=-1)                     # (B, N)

    labels_pos = labels.clamp(min=0)
    label_sum = labels_pos.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target = labels_pos / label_sum                               # (B, N)

    loss_per_query = -(target * log_probs).sum(dim=-1)            # (B,)

    valid = mask.sum(dim=-1) >= 2
    if valid.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return loss_per_query[valid].mean()


def listwise_ranknet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise RankNet loss aggregated across all pairs in each query group.

    For every pair (i, j) where label_i > label_j:
      loss += log(1 + exp(-(s_i - s_j)))
    """
    B, N = scores.shape
    s_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)         # (B, N, N)
    l_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)

    valid_pairs = (l_diff > 0) & mask.unsqueeze(-1) & mask.unsqueeze(-2)
    if valid_pairs.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    loss = torch.log1p(torch.exp(-s_diff)) * valid_pairs.float()
    return loss.sum() / valid_pairs.sum()


def listwise_lambdarank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    LambdaRank loss: pairwise logistic loss weighted by ΔNDCG.

    Pairs whose swap would change NDCG more get higher weight.
    """
    B, N = scores.shape

    s_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    l_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
    valid_pairs = (l_diff > 0) & mask.unsqueeze(-1) & mask.unsqueeze(-2)

    if valid_pairs.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Ranks from current scores
    _, inv = scores.sort(dim=-1, descending=True)
    ranks = torch.zeros_like(scores)
    for b in range(B):
        ranks[b, inv[b]] = torch.arange(N, device=scores.device, dtype=torch.float32)

    # ΔNDCG weight
    gains = (2.0 ** labels - 1).clamp(min=0)                      # (B, N)
    gain_diff = (gains.unsqueeze(-1) - gains.unsqueeze(-2)).abs()

    discount = 1.0 / torch.log2(ranks + 2.0)
    disc_diff = (discount.unsqueeze(-1) - discount.unsqueeze(-2)).abs()

    delta_ndcg = gain_diff * disc_diff

    losses = torch.log1p(torch.exp(-s_diff)) * delta_ndcg * valid_pairs.float()
    return losses.sum() / valid_pairs.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RerankerTrainer:
    def __init__(
        self,
        model: Qwen3Reranker,
        tokenizer: AutoTokenizer,
        train_dataset: RerankDataset,
        output_dir: str = "./output",
        val_dataset: Optional[RerankDataset] = None,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        loss_type: str = "listwise_softmax",
        temperature: float = 1.0,
        use_bf16: bool = True,
        log_interval: int = 10,
        save_interval: int = 500,
        eval_interval: int = 500,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.loss_type = loss_type
        self.temperature = temperature

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.use_amp = use_bf16 and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=True,
            )

        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.loss_fns = {
            "listwise_softmax": listwise_softmax_loss,
            "listwise_ranknet": listwise_ranknet_loss,
            "listwise_lambdarank": listwise_lambdarank_loss,
        }
        if loss_type not in self.loss_fns:
            raise ValueError(f"Unknown loss_type: {loss_type}. "
                             f"Choose from {list(self.loss_fns.keys())}")

        self.global_step = 0
        self.best_val_loss = float("inf")
        os.makedirs(output_dir, exist_ok=True)

    def _compute_loss(self, scores, labels, doc_mask):
        fn = self.loss_fns[self.loss_type]
        kwargs = {}
        if self.loss_type == "listwise_softmax":
            kwargs["temperature"] = self.temperature
        return fn(scores, labels, doc_mask, **kwargs)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            doc_mask = batch["doc_mask"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                scores = self.model(input_ids, attention_mask)
                loss = self._compute_loss(scores, labels, doc_mask)
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1

            if step % self.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss {loss.item() * self.gradient_accumulation_steps:.4f} | "
                    f"LR {self.scheduler.get_last_lr()[0]:.2e}"
                )

            if self.global_step > 0 and self.global_step % self.save_interval == 0:
                self._save(f"step_{self.global_step}")

            if (self.global_step > 0
                    and self.global_step % self.eval_interval == 0
                    and self.val_loader):
                val_loss = self.validate()
                logger.info(f"Step {self.global_step} | Val Loss {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best")
                self.model.train()

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            doc_mask = batch["doc_mask"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                scores = self.model(input_ids, attention_mask)
                loss = self._compute_loss(scores, labels, doc_mask)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def train(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Device: {self.device}")
        logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")
        logger.info(f"Loss: {self.loss_type}")
        logger.info(f"AMP: {self.use_amp} ({self.amp_dtype})")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} done | Avg train loss {train_loss:.4f}")

            if self.val_loader:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch} | Val loss {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best")

            self._save(f"epoch_{epoch}")

        self._save("final")
        logger.info("Training complete.")

    def _save(self, name: str):
        path = os.path.join(self.output_dir, name)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def rerank(
    model: Qwen3Reranker,
    tokenizer: AutoTokenizer,
    query: str,
    documents: List[str],
    max_length: int = 512,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> List[Tuple[int, float]]:
    """
    Score and rank documents for a query. Returns (doc_index, score) sorted
    by score descending.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    all_scores = []

    for i in range(0, len(documents), batch_size):
        chunk = documents[i:i + batch_size]
        ids_list, mask_list = [], []

        for doc in chunk:
            text = f"Query: {query}\nDocument: {doc}"
            enc = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids_list.append(enc["input_ids"].squeeze(0))
            mask_list.append(enc["attention_mask"].squeeze(0))

        input_ids = torch.stack(ids_list).unsqueeze(0).to(device)       # (1, N, L)
        attention_mask = torch.stack(mask_list).unsqueeze(0).to(device)

        batch_scores = model(input_ids, attention_mask).squeeze(0).cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        all_scores.extend(batch_scores)

    return sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Qwen3 4B reranker with Jina Rerank v3 listwise loss"
    )

    # Model
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--tokenizer_name", default=None,
                   help="Tokenizer path (defaults to model_name)")

    # Data
    p.add_argument("--train_data", required=True, help="JSONL training data")
    p.add_argument("--val_data", default=None, help="JSONL validation data")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_docs_per_query", type=int, default=20)

    # Training
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--loss_type", default="listwise_softmax",
                   choices=["listwise_softmax", "listwise_ranknet", "listwise_lambdarank"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_bf16", action="store_true", help="Disable bfloat16 AMP")

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=500)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer_name = args.tokenizer_name or args.model_name
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {args.model_name}")
    model = Qwen3Reranker(model_name=args.model_name)

    train_dataset = RerankDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_docs_per_query=args.max_docs_per_query,
    )

    val_dataset = None
    if args.val_data:
        val_dataset = RerankDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_docs_per_query=args.max_docs_per_query,
        )

    trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        temperature=args.temperature,
        use_bf16=not args.no_bf16,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
    )

    trainer.train()


if __name__ == "__main__":
    main()
