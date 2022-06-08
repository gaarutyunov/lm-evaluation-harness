"""
StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow
http://web.cse.ohio-state.edu/~sun.397/docs/StaQC-www18.pdf

StaQC (Stack Overflow Question-Code pairs) is the largest dataset to date of around 148K Python and 120K SQL domain
question-code pairs, which are automatically mined from Stack Overflow using a Bi-View Hierarchical Neural Network,
as described in the paper "StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow" (WWW'18).

Homepage: https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset
"""
import inspect

import lm_eval.datasets.conala.conala
from lm_eval.base import Task, rf
from lm_eval.metrics import bleu


class CoNaLa(Task):
    DATASET_NAME = None
    DATASET_PATH = inspect.getfile(lm_eval.datasets.conala.conala)

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
        return doc["question"]

    def doc_to_text(self, doc):
        return f'QUESTION:\n{doc["question"]}\nANSWER:\n'

    def doc_to_target(self, doc):
        return doc["answer"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, ["<|endoftext|>"])

    def process_results(self, doc, results):
        return {
            "bleu": (doc['answer'], results[0]),
        }

    def aggregation(self):
        return {
            "bleu": bleu,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
        }
