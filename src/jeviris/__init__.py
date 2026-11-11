"""Evaluate Jev on held-out Iris flowers through OpenRouter's Decisions API."""

import argparse
import asyncio
import json
import os
import re
import time
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

MODEL = "~typesafe/jev-latest"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
SPECIES = ["setosa", "versicolor", "virginica"]
FEATURES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
TREATMENTS = ["zero_shot", "few_shot", "full_train"]


# The API envelopes are dynamic JSON; predictions have a stable local shape.
type JSONObject = dict[str, Any]
type Measurements = Sequence[float] | NDArray[np.float64]


class Prediction(TypedDict, total=False):
    species: str | None
    probabilities: dict[str, float] | None
    confidence: float | None
    routed_model: str
    usage: JSONObject | None
    raw: JSONObject
    error: str
    elapsed_seconds: float
    attempts: int


class RunArgs(argparse.Namespace):
    treatments: list[str]
    shots: int
    test_size: float
    limit: int | None
    concurrency: int
    seed: int
    model: str
    output_dir: Path
    dry_run: bool


def flower(x: Measurements) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURES, x, strict=True)}


def build_request(
    treatment: str, x: Measurements, train_X: NDArray[np.float64],
    train_y: NDArray[np.int64], shots: int = 3, model: str = MODEL,
) -> JSONObject:
    if treatment == "zero_shot":
        indices: Sequence[int] = []
    elif treatment == "few_shot":
        indices = [i for label in range(3)
                   for i in [j for j, y in enumerate(train_y) if y == label][:shots]]
    elif treatment == "full_train":
        indices = range(len(train_y))
    else:
        raise ValueError(f"Unknown treatment: {treatment}")
    state: JSONObject = {"flower": flower(x)}
    if treatment != "zero_shot":
        state["labeled_examples"] = [
            {"flower": flower(train_X[i]), "species": SPECIES[train_y[i]]}
            for i in indices
        ]
    instructions = "Classify `flower` as an Iris species using its measurements in centimeters."
    if indices:
        instructions += " Use `labeled_examples` as reference observations."
    return {
        "model": model,
        "state": state,
        "questions": {"species": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {s: f"The flower is Iris {s}." for s in SPECIES},
        }},
    }


def parse_response(data: JSONObject) -> Prediction:
    model = data.get("model")
    if not isinstance(model, str) or not re.fullmatch(r"~?typesafe/jev-(?:latest|\d[\w.-]*)", model):
        raise ValueError(f"Unexpected response model: {model!r}")
    answer = data["answers"]["species"]
    if answer.get("type") != "choice" or answer.get("choice") not in SPECIES:
        raise ValueError(f"Invalid species answer: {answer!r}")
    return {
        "species": answer["choice"],
        "probabilities": answer.get("probabilities"),
        "confidence": answer.get("confidence"),
        "routed_model": model,
        "usage": data.get("usage"),
        "raw": data,
    }


async def classify(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: JSONObject,
    retries: int = 3, label: str = "request",
) -> Prediction:
    if retries < 1:
        raise ValueError("retries must be positive")
    async with sem:
        started = time.perf_counter()
        for attempt in range(retries):
            try:
                response = await client.post(ENDPOINT, json=payload)
                response.raise_for_status()
                result = parse_response(response.json())
                result["elapsed_seconds"] = time.perf_counter() - started
                result["attempts"] = attempt + 1
                return result
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                transient = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and (exc.response.status_code == 429 or exc.response.status_code >= 500)
                )
                if transient and attempt + 1 < retries:
                    print(f"  {label}: transient {type(exc).__name__}; "
                          f"retry {attempt + 1}/{retries - 1} in {2**attempt}s", flush=True)
                    await asyncio.sleep(2**attempt)
                    continue
                if isinstance(exc, httpx.HTTPStatusError):
                    # Never include request headers (and hence credentials) in output.
                    error = f"HTTP {exc.response.status_code}: {exc.response.text[:1000]}"
                else:
                    error = str(exc)
                return {"species": None, "error": error,
                        "elapsed_seconds": time.perf_counter() - started,
                        "attempts": attempt + 1}
    raise AssertionError("Unreachable retry state")


async def run(args: RunArgs) -> None:
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and not args.dry_run:
        raise SystemExit("Set OPENROUTER_API_KEY in .env or your environment.")
    iris = load_iris()
    train_indices, test_indices = train_test_split(
        list(range(len(iris.target))), test_size=args.test_size,
        stratify=iris.target, random_state=args.seed,
    )
    if args.limit is not None:
        test_indices = test_indices[:args.limit]
    train_X, train_y = iris.data[train_indices], iris.target[train_indices]
    payloads = {
        treatment: [build_request(treatment, iris.data[i], train_X, train_y,
                                  args.shots, args.model) for i in test_indices]
        for treatment in args.treatments
    }
    if args.dry_run:
        print(json.dumps({t: p[0] for t, p in payloads.items()}, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    summary: JSONObject = {"requested_model": args.model, "seed": args.seed,
               "test_size": args.test_size, "shots_per_class": args.shots,
               "train_indices": train_indices, "test_indices": test_indices,
               "treatments": {}}
    print(f"Model: {args.model} | seed: {args.seed} | concurrency: {args.concurrency}", flush=True)
    print(f"Split: {len(train_indices)} training rows, {len(test_indices)} test rows scored", flush=True)
    print(f"Test classes: {dict(Counter(SPECIES[iris.target[i]] for i in test_indices))}", flush=True)
    print(f"Planned requests: {sum(map(len, payloads.values()))} | output: {args.output_dir.resolve()}", flush=True)
    failures = 0
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"}, timeout=60,
    ) as client:
        for treatment, requests in payloads.items():
            started = time.perf_counter()
            examples = len(requests[0]["state"].get("labeled_examples", []))
            print(f"\n{treatment}: starting {len(requests)} requests, {examples} labeled examples each", flush=True)
            results = await asyncio.gather(*(
                classify(client, sem, payload, label=f"{treatment} row {i}")
                for i, payload in zip(test_indices, requests)
            ))
            correct = sum(r["species"] == SPECIES[iris.target[i]]
                          for r, i in zip(results, test_indices))
            failed = sum(r["species"] is None for r in results)
            failures += failed
            stats: JSONObject = {"correct": correct, "total": len(results), "failed": failed,
                     "accuracy": correct / len(results),
                     "answered_by": dict(Counter(r.get("routed_model", "failed") for r in results))}
            stats["elapsed_seconds"] = time.perf_counter() - started
            stats["cost"] = sum((r.get("usage") or {}).get("cost", 0) or 0 for r in results)
            summary["treatments"][treatment] = stats
            print(f"{treatment}: {correct}/{len(results)} = {stats['accuracy']:.1%} ({failed} failed)")
            print(f"  Resolved models: {stats['answered_by']}")
            print(f"  Wall time: {stats['elapsed_seconds']:.2f}s | reported cost: ${stats['cost']:.6f}")
            print("  Confusion matrix (true rows / predicted columns; failures separate):")
            print(f"  {'true':12s} " + " ".join(f"{s:>10s}" for s in SPECIES) + "     failed")
            for species in SPECIES:
                counts = Counter(r["species"] for r, i in zip(results, test_indices)
                                 if SPECIES[iris.target[i]] == species)
                print(f"  {species:12s} " + " ".join(f"{counts[s]:10d}" for s in SPECIES)
                      + f" {counts[None]:10d}")
            for error, n in Counter(r["error"] for r in results if "error" in r).items():
                print(f"  error x{n}: {error}")
            with (args.output_dir / f"{treatment}.jsonl").open("w") as f:
                for i, request, result in zip(test_indices, requests, results):
                    f.write(json.dumps({"row_index": i, "x": iris.data[i].tolist(),
                                        "true": SPECIES[iris.target[i]],
                                        "request": request, **result}) + "\n")
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"  Saved {args.output_dir / f'{treatment}.jsonl'} and {args.output_dir / 'summary.json'}", flush=True)
            if failed == len(results):
                raise SystemExit("All requests failed; stopping. See saved results for errors.")
    if failures:
        raise SystemExit("Some requests failed; see saved results for errors.")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--treatments", nargs="+", default=TREATMENTS, choices=TREATMENTS)
    p.add_argument("--shots", type=int, default=3, help="examples per class for few_shot")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--limit", type=int, help="only score the first N test rows")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--dry-run", action="store_true", help="print one request per treatment without calling the API")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args(namespace=RunArgs())
    if args.shots < 1 or args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        p.error("shots, concurrency, and limit must be positive")
    if not 0.02 <= args.test_size <= 0.98:
        p.error("test-size must be between 0.02 and 0.98 (at least three rows per split)")
    asyncio.run(run(args))
