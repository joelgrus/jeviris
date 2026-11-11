# Jeviris

Classify the classic Iris dataset with **Jev itself**, through OpenRouter's
[Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).
Jev is a decision model: requests contain `state` and a typed `choice` question,
and responses contain a species and probabilities. It cannot use Chat Completions.
`typesafe/jev-router` is a different product that selects other models.

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
cp .env.example .env
# Edit .env and set your OpenRouter API key.
uv run jeviris --limit 3
```

The smoke test makes nine billable requests: three flowers under each treatment.
Run the full experiment with `uv run jeviris` (135 requests by default).
For a smaller experiment: `uv run jeviris --treatments few_shot --limit 5`.
To inspect request bodies without a key or network:

```bash
uv run jeviris --dry-run
```

`python -m jeviris` also works after installation. Use `--help` for all options.
The default model is `~typesafe/jev-latest`; use `--model typesafe/jev-1.14`
to select a version explicitly. The resolved model is recorded in every result.

The experiment uses a seeded, stratified 105/45 train/test split. All treatments
use the same held-out rows. `zero_shot` supplies only the flower's measurements;
`few_shot` supplies three training examples per class; `full_train` supplies all
105 training rows. Training examples follow the deterministic shuffled split
order. No held-out labels are sent to Jev. Examples are context, not model training.
These experiments measure Jev's predictions; they do not guarantee high accuracy.

During a run, the terminal shows the model, split, class counts, planned requests,
and output directory. Retry messages appear immediately. Each treatment ends with
accuracy, a confusion matrix, resolved models, elapsed time, reported API cost,
and saved paths. Individual predictions and probabilities are available in the
saved results.

Results are saved in `results/<treatment>.jsonl`, including dataset row indices,
measurements, ground truth, request bodies, predictions, probabilities, raw
responses, resolved models, usage, latency, and attempt counts. `results/summary.json` records split indices,
configuration and accuracy. Failed requests count as incorrect and are reported
separately. The process exits unsuccessfully on any API or parsing failures and
stops after a treatment if every request fails. Only network errors, rate limits,
and server errors are retried (three attempts maximum). Each run overwrites its
selected treatment files and summary; use `--output-dir results/my-run` to keep runs.

Run the offline tests (mocked HTTP, no credentials or API charges):

```bash
uv run python -m unittest discover -s tests -v
```
