"""
A parallel corpus of Python functions and documentation strings for automated code documentation and code generation
https://arxiv.org/pdf/1707.02275.pdf

Preprocessed Python functions and docstrings for automated code documentation (code2doc) and
automated code generation (doc2code) tasks.

Homepage: https://github.com/EdinburghNLP/code-docstring-corpus
"""
import inspect

import lm_eval.datasets.code_docstring.code_docstring
from lm_eval.base import Task, rf
from lm_eval.metrics import bleu


class CodeDocstring(Task):
    DATASET_NAME = None
    DATASET_PATH = inspect.getfile(lm_eval.datasets.code_docstring.code_docstring)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["val"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_decontamination_query(self, doc):
        return doc["docstring"]

    def doc_to_text(self, doc):
        return f'QUESTION:\n{doc["docstring"]}\nANSWER:\n'

    def doc_to_target(self, doc):
        return doc["body"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, ["<|endoftext|>"])

    def process_results(self, doc, results):
        return {
            "bleu": (self.doc_to_target(doc), results[0]),
        }

    def aggregation(self):
        return {
            "bleu": bleu,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
        }


class CodeDocstringFunctions(CodeDocstring):
    VERSION = 1
    DATASET_NAME = "functions"


class CodeDocstringMethods(CodeDocstring):
    VERSION = 1
    DATASET_NAME = "methods"
