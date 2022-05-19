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
"""APPS dataset."""

import json
import os
import pathlib

import datasets
from datasets import DownloadManager, DatasetInfo

from .reindent import reindent_code

_CITATION = """\
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""

_DESCRIPTION = """\
The Automated Programming Progress Standard, abbreviated APPS, dataset consists of 10,000 coding
problems in total, with 131,777 test cases for checking solutions and 232,421 ground-truth solutions
written by humans. The APPS benchmark attempts to mirror how humans programmers are
evaluated by posing coding problems in unrestricted natural language and using test cases to evaluate
solution correctness. The problems range in difficulty from introductory to collegiate competition
level and measure coding and problem-solving ability.
"""

_HOMEPAGE = "https://github.com/hendrycks/apps"

_URLS = "https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz"

_LICENSE = "MIT License"

_LEVELS = [
    "introductory",
    "interview",
    "competition"
]


class HendrycksAPPS(datasets.GeneratorBasedBuilder):
    """APPS is a dataset consisting of 10,000 coding problems"""

    VERSION = datasets.Version("0.0.2")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="apps", version=VERSION, description="APPS is a dataset consisting of 10,000 "
                                                                         "coding problems")
    ]

    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "solution": datasets.Value("string"),
                "starter_code": datasets.Value("string"),
                "difficulty": datasets.Value("string"),
                "type": datasets.Value("string"),
                "in_outs": datasets.Value("string"),
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
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "APPS", "train"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "APPS", "test"
                    ),
                },
            ),
        ]

    def _generate_examples(self, basepath):
        for problem_path in sorted(pathlib.Path(basepath).iterdir()):
            test_case_path = os.path.join(basepath, problem_path, "input_output.json")
            meta_fname = os.path.join(basepath, problem_path, "metadata.json")
            question_fname = os.path.join(basepath, problem_path, "question.txt")
            sols_fname = os.path.join(basepath, problem_path, "solutions.json")
            starter_code = os.path.join(basepath, problem_path, "starter_code.py")

            if os.path.exists(starter_code):
                answer_type = "\nUse Call-Based format\n"
            else:
                answer_type = "\nUse Standard Input format\n"

            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)) or (not os.path.isfile(test_case_path)):
                continue

            if os.path.isfile(starter_code):
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()

            # Read the question description
            with open(test_case_path, 'r') as f:
                in_outs = f.read()

            # Read the question level
            with open(meta_fname, 'r') as f:
                difficulty_json = json.load(f)
                difficulty = difficulty_json["difficulty"]

            # Read all the solutions
            with open(sols_fname, 'r') as f:
                sols_str_list = json.load(f)
                for i, sol_str in enumerate(sols_str_list):
                    sol_str = reindent_code(sol_str)
                    yield problem_path.name + "-" + str(i), {
                        "question": question_str,
                        "solution": sol_str,
                        "starter_code": starter_code,
                        "type": answer_type,
                        "difficulty": difficulty,
                        "in_outs": in_outs
                    }

