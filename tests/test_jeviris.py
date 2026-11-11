import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from sklearn.datasets import load_iris

import jeviris


def answer(species: str = "setosa") -> jeviris.JSONObject:
    return {"model": "typesafe/jev-1.14-20261016", "answers": {
        "species": {"type": "choice", "choice": species,
                    "probabilities": {s: float(s == species) for s in jeviris.SPECIES},
                    "confidence": 1.0}}, "usage": {"cost": 0.00001}}


class Requests(unittest.TestCase):
    def test_treatments_and_no_test_label(self) -> None:
        iris = load_iris()
        for treatment, count in [("zero_shot", 0), ("few_shot", 9), ("full_train", 149)]:
            request = jeviris.build_request(treatment, iris.data[0], iris.data[1:], iris.target[1:])
            self.assertEqual(request["questions"]["species"]["type"], "choice")
            self.assertEqual(len(request["state"].get("labeled_examples", [])), count)
            self.assertNotIn("species", request["state"]["flower"])

    def test_reject_wrong_model_and_invalid_choice(self) -> None:
        for model, species in [("typesafe/jev-router", "setosa"), ("openai/gpt-6", "setosa"),
                               ("typesafe/jev-1.14", "unknown")]:
            data = answer(species)
            data["model"] = model
            with self.assertRaises(ValueError):
                jeviris.parse_response(data)


class API(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_contract(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), jeviris.ENDPOINT)
            self.assertEqual(request.headers["Authorization"], "Bearer test")
            self.assertIn("questions", json.loads(request.content))
            return httpx.Response(200, json=answer())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle),
                                     headers={"Authorization": "Bearer test"}) as client:
            result = await jeviris.classify(client, asyncio.Semaphore(1), {"questions": {}})
        self.assertEqual(result["species"], "setosa")
        self.assertEqual(result["confidence"], 1)

    async def test_bad_request_not_retried(self) -> None:
        calls: list[httpx.Request] = []
        def handle(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await jeviris.classify(client, asyncio.Semaphore(1), {})
        self.assertEqual(len(calls), 1)
        self.assertIsNone(result["species"])
        self.assertIn("HTTP 400", result["error"])

    async def test_transient_failure_retried(self) -> None:
        calls: list[httpx.Request] = []
        def handle(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(503 if len(calls) == 1 else 200, json=answer())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with patch("jeviris.asyncio.sleep", return_value=None):
                result = await jeviris.classify(client, asyncio.Semaphore(1), {})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["species"], "setosa")

    async def test_end_to_end_artifacts(self) -> None:
        original_client = httpx.AsyncClient
        def client(**kwargs: object) -> httpx.AsyncClient:
            return original_client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=answer())), **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            args = jeviris.parser().parse_args(["--limit", "3", "--output-dir", directory])
            with patch.dict(os.environ, OPENROUTER_API_KEY="test"), \
                 patch("jeviris.httpx.AsyncClient", side_effect=client), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                await jeviris.run(args)
            for expected in ["Planned requests: 9",
                             "Confusion matrix", "Resolved models", "reported cost", "Saved"]:
                self.assertIn(expected, output.getvalue())
            self.assertNotIn("predicted=", output.getvalue())
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertFalse(set(summary["train_indices"]) & set(summary["test_indices"]))
            for treatment in jeviris.TREATMENTS:
                rows = [json.loads(line) for line in (Path(directory) / f"{treatment}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 3)
                self.assertEqual(summary["treatments"][treatment]["failed"], 0)
                self.assertEqual([r["row_index"] for r in rows], summary["test_indices"])


if __name__ == "__main__":
    unittest.main()
