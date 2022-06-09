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
"""code-docstring dataset."""
from lib2to3.refactor import RefactoringTool, get_fixers_from_package

import datasets
from datasets import DownloadManager, DatasetInfo

from lm_eval.code_utils import reindent_code

_DESCRIPTION = """\
Preprocessed Python functions and docstrings for automated code documentation (code2doc) and 
automated code generation (doc2code) tasks. 
"""

_CITATION = """\
@article{barone2017parallel,
  title={A parallel corpus of python functions and documentation strings for automated code documentation and code generation},
  author={Barone, Antonio Valerio Miceli and Sennrich, Rico},
  journal={arXiv preprint arXiv:1707.02275},
  year={2017}
}
"""

_HOMEPAGE = "https://github.com/EdinburghNLP/code-docstring-corpus"

_LICENSE = ""

_URLS = {
    "functions": {
        "train": {
            "bodies": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_bodies.train",
            "docstrings": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_desc.train",
            "declarations": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_decl.train",
        },
        "test": {
            "bodies": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_bodies.test",
            "docstrings": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_desc.test",
            "declarations": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_decl.test",
        },
        "valid": {
            "bodies": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_bodies.valid",
            "docstrings": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_desc.valid",
            "declarations": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_decl.valid",
        },
    },
    "methods": {
        "train": {
            "bodies": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_bodies.train.gz",
            "docstrings": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_desc.train",
            "declarations": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_desc.train",
        },
        "test": {
            "bodies": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_bodies.test",
            "docstrings": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_desc.test",
            "declarations": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_desc.test",
        },
        "valid": {
            "bodies": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_bodies.valid",
            "docstrings": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_desc.valid",
            "declarations": "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split/repo_split.parallel_methods_desc.valid",
        },
    },
}

_NAMES = ["functions", "methods"]


class CodeDocstring(datasets.GeneratorBasedBuilder):
    """A parallel corpus of python functions and documentation strings for automated code documentation and code
    generation"""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
            version=version,
            description="A parallel corpus of python functions and documentation strings for automated code "
            "documentation and code generation",
        )
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    refactor = RefactoringTool(fixer_names=get_fixers_from_package("lib2to3.fixes"))

    def _generate_examples(self, bodies, docstrings, declarations):
        with (
            open(bodies, encoding="latin-1") as bodies_f,
            open(docstrings, encoding="latin-1") as docstrings_f,
            open(declarations, encoding="latin-1") as declarations_f,
        ):
            i = 0
            for body, docstring, declaration in zip(
                bodies_f.readlines(),
                docstrings_f.readlines(),
                declarations_f.readlines(),
            ):
                docstring = docstring.strip().strip("\n").strip('"').strip("'")

                if len(docstring) == 0:
                    continue

                try:
                    body = (
                        body.replace(" DCNL DCSP ", "\n\t")
                        .replace(" DCSP ", "\t")
                        .replace(" DCNL ", "\n")
                    )
                    declaration = (
                        declaration.replace(" DCNL DCSP ", "\n\t")
                        .replace(" DCSP ", "\t")
                        .replace(" DCNL ", "\n")
                    )
                    docstring = (
                        docstring.replace(" DCNL DCSP ", "\n\t")
                        .replace(" DCSP ", "\t")
                        .replace(" DCNL ", "\n")
                    )
                    body = body.strip(" ").strip("\n")
                    body = declaration + body + "\n"
                    body = self.refactor.refactor_string(body, declaration)
                    func = (
                        str(body).lstrip(declaration).replace("\n\t", "\n").lstrip("\t")
                    )
                    reindented = reindent_code(func)
                    declaration = declaration.rstrip("\n")
                except Exception as e:
                    continue

                yield str(i), {
                    "body": reindented,
                    "docstring": docstring,
                    "declaration": declaration,
                }

                i += 1

    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {
                "docstring": datasets.Value("string"),
                "body": datasets.Value("string"),
                "declaration": datasets.Value("string"),
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
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs=data_dir["train"]
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs=data_dir["test"]
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs=data_dir["valid"]
            ),
        ]
