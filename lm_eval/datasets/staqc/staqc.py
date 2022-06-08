# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""StaQC dataset."""
from io import StringIO

import datasets
import numpy as np
import pandas as pd
from datasets import DownloadManager, DatasetInfo

from lm_eval.code_utils import repair_program_io, reindent_code

_CITATION = """\
@inproceedings{yao2018staqc,
  title={StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow},
  author={Yao, Ziyu and Weld, Daniel S and Chen, Wei-Peng and Sun, Huan},
  booktitle={Proceedings of the 2018 World Wide Web Conference on World Wide Web},
  pages={1693--1703},
  year={2018},
  organization={International World Wide Web Conferences Steering Committee}
}
"""

_DESCRIPTION = """\
StaQC (Stack Overflow Question-Code pairs) is the largest dataset to date of around 148K Python and 120K SQL domain
question-code pairs, which are automatically mined from Stack Overflow using a Bi-View Hierarchical Neural Network,
as described in the paper "StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow" (WWW'18).
"""

_URLS = {
    "python": {
        "question_code_pairs": "https://raw.githubusercontent.com/LittleYUYU/StackOverflow-Question-Code-Dataset/master/final_collection/python_multi_code_iids.txt",
        "multi_code_snippet": "https://raw.githubusercontent.com/LittleYUYU/StackOverflow-Question-Code-Dataset/master/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle",
        "multi_code_questions": "https://raw.githubusercontent.com/LittleYUYU/StackOverflow-Question-Code-Dataset/master/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle",
        "single_code_snippet": "https://raw.githubusercontent.com/LittleYUYU/StackOverflow-Question-Code-Dataset/master/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle",
        "single_code_questions": "https://raw.githubusercontent.com/LittleYUYU/StackOverflow-Question-Code-Dataset/master/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle",
    }
}

_HOMEPAGE = "https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset"

_LICENSE = "cc-by-4.0"

_NAMES = [
    "python",
    # "sql"
]


class StaQC(datasets.GeneratorBasedBuilder):
    """StaQC: a systematically mined dataset containing around 148K Python and 120K SQL domain question-code pairs"""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
            version=version,
            description="StaQC: a systematically mined dataset containing around 148K Python and 120K SQL domain "
            "question-code pairs",
        )
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    def _generate_examples(
        self,
        question_code_pairs: dict,
        multi_code_snippet: dict,
        multi_code_questions: dict,
        single_code_snippet: dict,
        single_code_questions: dict,
    ):
        for question_id, code_idx in question_code_pairs.items():
            code = multi_code_snippet[(question_id, code_idx)]
            question = multi_code_questions[question_id]

            code, _ = repair_program_io(code)
            code = reindent_code(code)

            yield question_id, {"question": question, "answer": code}

        for question_id, question, code in zip(
            single_code_questions.keys(),
            single_code_questions.values(),
            single_code_snippet.values(),
        ):
            code, _ = repair_program_io(code)
            code = reindent_code(code)

            yield question_id, {"question": question, "answer": code}

    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        config = self.config.name
        urls: dict = _URLS[config]
        data_dir = dl_manager.download_and_extract(urls)

        def read_txt(filename):
            with open(filename) as f:
                content = f.read().replace("(", "").replace(")", "")

            s = StringIO(content)

            return dict(
                np.genfromtxt(s, names=["question_id", "code_snippet_idx"], dtype="i4,i4",  delimiter=", ")
            )

        def file_to_df(key: str) -> pd.DataFrame:
            filename = data_dir[key]

            return (
                pd.read_pickle(filename) if key != "question_code_pairs" else read_txt(filename)
            )

        kwargs = dict(zip(urls.keys(), map(file_to_df, urls.keys())))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=kwargs,
            ),
        ]
