"""
Description2Code Dataset

Description2Code is dataset of ~7764 programming challenges scraped from CodeChef, Codeforces, & Hackerearth in 2016.
Each programing challenge in this dataset provides a problem description and multiple solutions and multiple test
cases.

Homepage: https://github.com/ethancaballero/description2code
"""
import inspect

import lm_eval.datasets.description2code.description2code
from lm_eval.base import Task, rf
from lm_eval.metrics import bleu


class Description2Code(Task):
    DATASET_NAME = None
    DATASET_PATH = inspect.getfile(lm_eval.datasets.description2code.description2code)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return NotImplemented

    def doc_to_decontamination_query(self, doc):
        return doc["problem"]

    def doc_to_text(self, doc):
        return f'QUESTION:\n{doc["problem"]}\nANSWER:\n'

    def doc_to_target(self, doc):
        return doc["solution"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, ["<|endoftext|>"])

    def process_results(self, doc, results):
        return {
            "bleu": (doc["solution"], results[0]),
        }

    def aggregation(self):
        return {
            "bleu": bleu,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
        }


class CodechefExternal(Description2Code):
    VERSION = 1
    DATASET_NAME = "codechef_external"


class CodechefHard(Description2Code):
    VERSION = 1
    DATASET_NAME = "codechef_hard"


class CodechefHarder(Description2Code):
    VERSION = 1
    DATASET_NAME = "codechef_harder"


class CodechefHardest(Description2Code):
    VERSION = 1
    DATASET_NAME = "codechef_hardest"


class CodechefMedium(Description2Code):
    VERSION = 1
    DATASET_NAME = "codechef_medium"


class CodechefEasy(Description2Code):
    VERSION = 1
    DATASET_NAME = "codechef_easy"


class HackerEarthNormal(Description2Code):
    VERSION = 1
    DATASET_NAME = "hackerearth_normal"


class HackerEarthCollege(Description2Code):
    VERSION = 1
    DATASET_NAME = "hackerearth_college"


