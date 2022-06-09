import pathlib
import pdb
from dataclasses import dataclass
from lib2to3.refactor import RefactoringTool, get_fixers_from_package

import datasets
import pandas as pd
from datasets import DownloadManager, DatasetInfo

from lm_eval.code_utils import reindent_code
from bs4 import BeautifulSoup


_DESCRIPTION = """\
CodeNet consists of over 14 million code samples and about 500 million lines of code in 55 
different programming languages, which is aimed at teaching AI to code. In addition to its large scale, CodeNet has a 
rich set of high-quality annotations to benchmark and help accelerate research in AI techniques for a variety of 
critical coding tasks, including code similarity and classification, code translation between a large variety of 
programming languages, and code performance (runtime and memory) improvement techniques. Additionally, 
CodeNet provides sample input and output test sets for 98.5% of the code samples, which can be used as an oracle for 
determining code correctness and potentially guide reinforcement learning for code quality improvements.
"""

_LICENSE = "apache"

_HOMEPAGE = "https://github.com/IBM/Project_CodeNet"

_CITATION = """\
@article{puri2021project,
  title={Project codenet: a large-scale AI for code dataset for learning a diversity of coding tasks},
  author={Puri, Ruchir and Kung, David S and Janssen, Geert and Zhang, Wei and Domeniconi, Giacomo and Zolotov, Vladmir
  and Dolby, Julian and Chen, Jie and Choudhury, Mihir and Decker, Lindsey and others},
  journal={ArXiv. Available at https://arxiv. org/abs},
  volume={2105},
  year={2021}
}
"""

_URLS = {
    "benchmarks": {
        "python": "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800.tar.gz",
        "java": "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Java250.tar.gz",
        "cpp1": "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_C++1000.tar.gz",
        "cpp2": "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_C++1400.tar.gz",
    },
    "metadata": "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_metadata.tar.gz",
    "problem_list": ""
}

_NAMES = [
    "python",
    # "java",
    # "cpp1",
    # "cpp2"
]

_NAME_TO_EXT = {"python": "py", "java": "java", "cpp1": "cc", "cpp2": "cc"}


@dataclass
class CodeNetBuilderConfig(datasets.BuilderConfig):
    extension: str = "py"


class CodeNet(datasets.GeneratorBasedBuilder):
    """Project codenet: a large-scale AI for code dataset for learning a diversity of coding tasks"""

    refactor = RefactoringTool(fixer_names=get_fixers_from_package("lib2to3.fixes"))

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        CodeNetBuilderConfig(
            name=name,
            version=version,
            description="Project codenet: a large-scale AI for code dataset for learning a diversity of coding tasks",
            extension=_NAME_TO_EXT[name],
        )
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    def _generate_examples(self, benchmark: str, metadata: str, problem_list: str):
        problems_df = pd.read_csv(problem_list, usecols=["id", "dataset"], index_col=0)
        mask = problems_df["dataset"] == "AIZU"

        for problem_id, _ in problems_df[mask].iterrows():
            problem_meta = pathlib.Path(metadata).joinpath("metadata", f'{problem_id}.csv')

            with open(
                pathlib.Path(metadata).joinpath(
                    "problem_descriptions", f"{problem_id}.html"
                )
            ) as f:
                problem_html = f.read()

            soup = BeautifulSoup(problem_html, features="html.parser")

            problem = soup.get_text()

            meta_df = pd.read_csv(
                str(problem_meta),
                usecols=["submission_id", "filename_ext", "status"],
                index_col=0,
            )
            mask = (meta_df["filename_ext"] == self.config.extension) & (
                meta_df["status"] == "Accepted"
            )

            for submission_id, row in meta_df[mask].iterrows():
                submission_file = f'{submission_id}.{row["filename_ext"]}'

                with open(
                    pathlib.Path(benchmark).joinpath(problem_id, submission_file)
                ) as f:
                    solution = f.read()
                    try:
                        solution = self.refactor.refactor_string(
                            solution, submission_id
                        )
                    except Exception as e:
                        pdb.set_trace()
                        continue

                    solution = reindent_code(str(solution))

                    yield f"{problem_id}-{submission_id}", {
                        "problem": problem,
                        "solution": solution,
                    }

    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {
                "problem": datasets.Value("string"),
                "solution": datasets.Value("string"),
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
        urls = {
            "benchmark": _URLS["benchmarks"][self.config.name],
            "metadata": _URLS["metadata"],
        }
        data_dir = dl_manager.download_and_extract(urls)

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=data_dir)]
