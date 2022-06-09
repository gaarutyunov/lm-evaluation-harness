"""
Project codenet: a large-scale AI for code dataset for learning a diversity of coding tasks
https://arxiv.org/pdf/2105.12655.pdf

CodeNet consists of over 14 million code samples and about 500 million lines of code in 55
different programming languages, which is aimed at teaching AI to code.

Homepage: https://github.com/IBM/Project_CodeNet
"""
import inspect

import lm_eval.datasets.codenet.codenet
from lm_eval.base import Task, rf
from lm_eval.metrics import bleu


class CodeNet(Task):
    DATASET_NAME = None
    DATASET_PATH = inspect.getfile(lm_eval.datasets.codenet.codenet)

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


class CodeNetPython(CodeNet):
    VERSION = 1
    DATASET_NAME = "python"
