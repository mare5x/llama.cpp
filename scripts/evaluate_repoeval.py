from __future__ import annotations  # for pyright

import argparse
import json
import random
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import editdistance
import requests
from tqdm import tqdm


class Model(ABC):
    @abstractmethod
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError


class ServerModel(Model):
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.request_kwargs = request_kwargs

    def __call__(self, prompt: str) -> str:
        request = {"prompt": prompt}
        if self.request_kwargs is not None:
            request.update(self.request_kwargs)

        resp = requests.post(f"{self.base_url}/completion", json=request)
        json_resp = resp.json()
        return json_resp["content"]


class CLIModel(Model):
    def __init__(self, bin_path: Path, bin_args: str | None = None) -> None:
        self.bin_path = bin_path
        self.bin_args = [] if bin_args is None else shlex.split(bin_args)

    def __call__(self, prompt: str) -> str:
        # ["-r", '\n'] doesn't work because `prompt` ends with `\n`
        cmd = [str(self.bin_path), "-p", prompt] + self.bin_args
        resp = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        content = resp.stdout[len(prompt) :]
        return content


def split_and_strip_lines(s: str) -> list[str]:
    out = []
    for line in s.splitlines():
        line = line.strip()
        if line:
            out.append(line)
    return out


def compute_exact_match(
    target: str, predictions: list[str], pass_at_k: int = 1
) -> float:
    target_lines = split_and_strip_lines(target)
    scores = []
    for prediction in predictions[:pass_at_k]:
        prediction_lines = split_and_strip_lines(prediction)[: len(target_lines)]
        scores.append(target_lines == prediction_lines)
    return 1.0 if any(scores) else 0.0


def compute_edit_similarity(
    target: str, predictions: list[str], pass_at_k: int = 1
) -> float:
    target_lines = split_and_strip_lines(target)
    target_str = "".join(target_lines)
    scores = []
    for prediction in predictions[:pass_at_k]:
        prediction_lines = split_and_strip_lines(prediction)[: len(target_lines)]
        prediction_str = "".join(prediction_lines)
        es = 1 - (
            editdistance.eval(target_str, prediction_str)
            / max(len(target_str), len(prediction_str))
        )
        scores.append(es)
    return max(scores)


def randomize_prompt_target(prompt: str, target: str) -> tuple[str, str]:
    first_char_pos = len(target) - len(target.lstrip())
    first_char_pos += random.Random(prompt).randint(0, len(target.lstrip()) - 1)
    new_prompt = prompt + target[:first_char_pos]
    new_target = target[first_char_pos:]
    return new_prompt, new_target


def compute_predictions(
    data_path: Path,
    out_path: Path,
    model: Model,
    pass_at_k: int = 1,
    randomize_target: bool = False,
) -> None:
    with open(data_path, "r", encoding="utf8") as f_in:
        n_examples = sum(1 for _ in f_in)

    with open(data_path, "r", encoding="utf8") as f_in, open(
        out_path, "w", encoding="utf8"
    ) as f_out:
        for line in tqdm(f_in, total=n_examples):
            example = json.loads(line)
            prompt = example["prompt"] + "\n"
            target = example["metadata"]["ground_truth"]
            if randomize_target:
                prompt, target = randomize_prompt_target(prompt, target)

            print(
                "\n".join(f">{line}" for line in prompt.rsplit("\n", maxsplit=4)[-4:])
            )
            print("-      expected:", repr(target))

            predictions = []
            for it in range(pass_at_k):
                predicted = model(prompt)
                predictions.append(predicted)
                print(f"+ got [{it + 1:02} / {pass_at_k:02}]:", repr(predicted))

            out = {"target": target, "predictions": predictions}
            out_json = json.dumps(out)
            f_out.write(out_json + "\n")


def compute_metrics(out_path: Path, pass_at_k: int = 1) -> None:
    metrics = {
        "exact_match": compute_exact_match,
        "edit_similarity": compute_edit_similarity,
    }
    out: dict[str, list[float]] = {k: [] for k in metrics}
    with open(out_path, "r", encoding="utf8") as f:
        for line in f:
            example = json.loads(line)
            for metric_name, metric_func in metrics.items():
                score = metric_func(
                    example["target"], example["predictions"], pass_at_k=pass_at_k
                )
                out[metric_name].append(score)
    for metric_name, scores in out.items():
        mean = 100 * sum(scores) / len(scores)
        print(f"{metric_name:16}: {mean:0.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", type=Path, help="Path to the RepoEval jsonl dataset."
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output file for storing predictions/calculating metrics.",
    )
    executor_group = parser.add_mutually_exclusive_group(required=False)
    executor_group.add_argument(
        "--cli", help="Make predictions with the path to the `llama-cli` executable."
    )
    executor_group.add_argument(
        "--server",
        nargs="?",
        const="http://localhost:8080",
        default="http://localhost:8080",
        help="Make predictions using `llama-server` running at the given url.",
    )
    parser.add_argument(
        "--args",
        help=""
        'If `--server`, JSON format server request overrides (e.g. \'{"token_healing": "d"}\'). '
        "If `--cli`, cli arguments (e.g. '-m model.gguf -c 1024.')",
    )
    parser.add_argument(
        "--passk",
        type=int,
        default=1,
        help="How many predictions to make for pass-at-k.",
    )
    parser.add_argument(
        "--randomize_target",
        action="store_true",
        help=""
        "Instead of having to complete '    return True', the new target will be e.g. "
        "'turn True' with '    re' added to the prompt.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Only calculate metrics for stored predictions.",
    )
    args = parser.parse_args()

    if not args.metrics:
        model: Model
        if args.cli:
            assert (
                args.args is not None
            ), "Provide the model to use with `--args '-m model.gguf'`"
            model = CLIModel(args.cli, bin_args=args.args)
        else:
            server_args = json.loads(args.args) if args.args else None
            model = ServerModel(args.server, request_kwargs=server_args)
        compute_predictions(
            args.data_path,
            args.output_path,
            model,
            pass_at_k=args.passk,
            randomize_target=args.randomize_target,
        )
    compute_metrics(args.output_path)
