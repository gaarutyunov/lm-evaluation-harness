"""
Measuring Coding Challenge Competence With APPS
https://arxiv.org/pdf/2105.09938

The Automated Programming Progress Standard, abbreviated APPS, dataset consists of 10,000 coding
problems in total, with 131,777 test cases for checking solutions and 232,421 ground-truth solutions
written by humans. The APPS benchmark attempts to mirror how humans programmers are
evaluated by posing coding problems in unrestricted natural language and using test cases to evaluate
solution correctness. The problems range in difficulty from introductory to collegiate competition
level and measure coding and problem-solving ability.

Homepage: https://github.com/hendrycks/apps
"""
import inspect

import lm_eval.datasets.hendrycks_apps.hendrycks_apps
from lm_eval.base import Task, rf
from lm_eval.code_utils import timeout_handler, run_test
from lm_eval.metrics import mean

import signal

import numpy as np


signal.signal(signal.SIGALRM, timeout_handler)


class Apps(Task):
    DATASET_NAME = None
    DATASET_PATH = inspect.getfile(lm_eval.datasets.hendrycks_apps.hendrycks_apps)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_decontamination_query(self, doc):
        return doc["question"]

    def doc_to_text(self, doc):
        return f'QUESTION:\n{doc["question"]}\n{doc["starter_code"]}\n{doc["type"]}\nANSWER:\n'

    def doc_to_target(self, doc):
        return doc["solution"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, ["<|endoftext|>"])

    def process_results(self, doc, results):
        test_cases = run_test(test=results[0], in_outs=doc["in_outs"])
        if len(test_cases) == 0:
            test_cases = [False]

        test_cases = np.array(test_cases)

        syntax_errors_rate = len(test_cases[test_cases == -2]) / len(test_cases)
        runtime_errors_rate = len(test_cases[test_cases == -1]) / len(test_cases)

        test_cases = test_cases > 0

        return {
            "strict_acc": np.all(test_cases).astype(float),
            "avg_test_cases": np.mean(test_cases).astype(float),
            "syntax_errors_rate": syntax_errors_rate,
            "runtime_errors_rate": runtime_errors_rate,
        }

    def aggregation(self):
        return {
            "strict_acc": mean,
            "avg_test_cases": mean,
            "syntax_errors_rate": mean,
            "runtime_errors_rate": mean,
        }

    def higher_is_better(self):
        return {
            "strict_acc": True,
            "avg_test_cases": True,
            "syntax_errors_rate": False,
            "runtime_errors_rate": False,
        }


class AppsIntroductory(Apps):
    VERSION = 1
    DATASET_NAME = "introductory"


class AppsInterview(Apps):
    VERSION = 1
    DATASET_NAME = "interview"


class AppsCompetition(Apps):
    VERSION = 1
    DATASET_NAME = "competition"
