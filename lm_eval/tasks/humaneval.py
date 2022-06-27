from asyncio import as_completed
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from human_eval.evaluation import estimate_pass_at_k
from human_eval.execution import check_correctness
from lm_eval.metrics import mean
from tqdm.auto import tqdm

from lm_eval.base import Task, rf


def evaluate_functional_correctness(
        doc: dict,
        completions: List[str],
        k=None,
        n_workers: int = 4,
        timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples
    """

    if k is None:
        k = [1, 10, 100]

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for completion in tqdm(completions):
            task_id = doc["task_id"]
            args = (doc, completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    return pass_at_k


class HumanEval(Task):
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return f"QUESTION:\n{doc['prompt']}\nANSWER:\n"

    def doc_to_target(self, doc):
        return doc["canonical_solution"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, ["<|endoftext|>"])

    def process_results(self, doc, results):
        return evaluate_functional_correctness(doc, results)

    def aggregation(self):
        return {
            "pass@1": mean,
            "pass@10": mean,
            "pass@100": mean,
        }

    def higher_is_better(self):
        return {
            "pass@1": True,
            "pass@10": True,
            "pass@100": True,
        }
