# train.py
import argparse
import torch
import numpy as np
import evaluate

from util import CharacterTokenizer, Dataset
from gpt import GPTLanguageModel


@torch.no_grad()
def estimate_loss(data, model, eval_iters=100):
    device = next(model.parameters()).device
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = data.get_batch(split, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class Metrics:
    """
    mode:
      - 'off'   : only perplexity
      - 'light' : perplexity, ROUGE-1/L, accuracy
      - 'full'  : perplexity, ROUGE-1/L, BERTScore, accuracy
    """
    def __init__(self, mode='full', number_of_steps=3, bert_model='roberta-base'):
        self.mode = mode
        self.number_of_steps = number_of_steps
        self.bertscore_model = bert_model

        self.rouge = None
        self.accuracy = None
        self.bertscore = None

        if self.mode in ('light', 'full'):
            self.rouge = evaluate.load("rouge")
            self.accuracy = evaluate.load("accuracy")
        if self.mode == 'full':
            self.bertscore = evaluate.load("bertscore")

        if self.mode == 'off':
            self.keys = ["perplexity"]
        elif self.mode == 'light':
            self.keys = ["perplexity", "rouge1", "rougeL", "accuracy"]
        else:
            self.keys = ["perplexity", "rouge1", "rougeL", "bertscore", "accuracy"]

    def step(self, data, model, tokenizer):
        device = next(model.parameters()).device

        # perplexity from validation loss
        x, y = data.get_batch('val', device)
        _, loss = model(x, y)
        perp = float(torch.exp(loss).item())

        if self.mode == 'off':
            return [perp]

        # generate continuations to compare with ground truth slice
        number_of_samples = data.context_size
        x2, y2 = data.get_batch('val', device, y_shift=number_of_samples)
        gen_x = model.generate(x2, number_of_samples)   # (B, T+N); we take last N
        gen_x = gen_x[:, -number_of_samples:]

        # decode for text metrics
        generated_texts = [tokenizer.decode(i) for i in gen_x.detach().cpu().numpy()]
        reference_texts = [tokenizer.decode(i) for i in y2.detach().cpu().numpy()]

        # ROUGE (fast)
        rouge_results = self.rouge.compute(predictions=generated_texts, references=reference_texts)
        rouge1 = float(rouge_results["rouge1"])
        rougeL = float(rouge_results["rougeL"])

        # token-level accuracy
        pred_flat = gen_x.detach().cpu().reshape(-1).tolist()
        ref_flat = y2.detach().cpu().reshape(-1).tolist()
        acc = float(self.accuracy.compute(predictions=pred_flat, references=ref_flat)["accuracy"])

        if self.mode == 'light':
            return [perp, rouge1, rougeL, acc]

        # BERTScore (heavier; default roberta-base, switch via arg)
        bs = self.bertscore.compute(
            predictions=generated_texts,
            references=reference_texts,
            lang="en",
            model_type=self.bertscore_model
        )
        bert_f1 = float(np.mean(bs["f1"]))

        return [perp, rouge1, rougeL, bert_f1, acc]

    @torch.no_grad()
    def __call__(self, data, model, tokenizer):
        model_was_training = model.training
        model.eval()
        try:
            rows = []
            for _ in range(self.number_of_steps):
                rows.append(self.step(data, model, tokenizer))
            agg = np.mean(np.array(rows), axis=0).tolist()
            return dict(zip(self.keys, agg))
        finally:
            model.train(model_was_training)


def train(data, model, tokenizer, steps, report_frequency, lr, metrics_mode, metric_steps, bertscore_model):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    metrics = Metrics(mode=metrics_mode, number_of_steps=metric_steps, bert_model=bertscore_model)

    for step in range(steps):
        xb, yb = data.get_batch('train', device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % report_frequency == 0 or step == steps - 1:
            losses = estimate_loss(data, model)
            print(f"Step {step}, train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")
            metrics_dict = metrics(data, model, tokenizer)
            print("Metrics:", metrics_dict)
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)  # used inside model components

    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--save", type=str, default="model.pth")
    train_parser.add_argument("--steps", type=int, default=5000)
    train_parser.add_argument("--report", type=int, default=500)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--metrics", choices=["off", "light", "full"], default="full")
    train_parser.add_argument("--metric-steps", type=int, default=3)
    train_parser.add_argument("--bertscore-model", type=str, default="roberta-base")

    # EVAL
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--load", type=str, default="model.pth")
    eval_parser.add_argument("--prompt", type=str)
    eval_parser.add_argument("--token-count", type=int, default=300)
    eval_parser.add_argument("--temperature", type=float, default=0.9)
    eval_parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: Running on cpu!")

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()
    tokenizer = CharacterTokenizer(content)
    data = torch.tensor(tokenizer.encode(content), dtype=torch.long)
    dataset = Dataset(data, args.context_size, args.batch_size)

    # Model
    model = GPTLanguageModel(
        vocab_size=len(tokenizer.vocab),
        n_embd=args.n_embd,
        context_size=args.context_size,
        n_head=args.n_head,
        n_layer=args.n_layer
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
    print(f"Using device: {device}\n")

    if args.command == "eval":
        print("=" * 20, "INFERENCE", "=" * 20)
        model.load_state_dict(torch.load(args.load, map_location=device))
        model.eval()

        # prepare context
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        if args.prompt is not None:
            context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
        max_tokens = args.token_count

        print(
            tokenizer.decode(
                model.generate(
                    start_idx=context,
                    number_of_tokens=max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )[0].tolist()
            )
        )

    elif args.command == "train":
        print("=" * 20, "TRAINING", "=" * 20)
        train(
            dataset,
            model,
            tokenizer,
            steps=args.steps,
            report_frequency=args.report,
            lr=args.lr,
            metrics_mode=args.metrics,
            metric_steps=args.metric_steps,
            bertscore_model=args.bertscore_model,
        )
        torch.save(model.state_dict(), args.save)
        print("=" * 50)

        # Optional: quick sample after training with safe defaults
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(
            tokenizer.decode(
                model.generate(
                    start_idx=context,
                    number_of_tokens=300,
                    temperature=0.9,
                    top_p=0.9
                )[0].tolist()
            )
        )


if __name__ == "__main__":
    main()
