import os.path
import pathlib
import pdb
from lib2to3.refactor import RefactoringTool, get_fixers_from_package

import datasets
from datasets import DownloadManager, DatasetInfo

from lm_eval.code_utils import reindent_code

_DESCRIPTION = """\
Description2Code is dataset of ~7764 programming challenges scraped from CodeChef, Codeforces, 
& Hackerearth in 2016. Each programing challenge in this dataset provides a problem description and multiple 
solutions and multiple test cases. 
"""

_LICENSE = ""

_CITATION = """\
@misc{Caballero_Description2Code_Dataset_2016,
    author = {Caballero, Ethan and OpenAI, . and Sutskever, Ilya},
    doi = {10.5281/zenodo.5665051},
    month = {8},
    title = {{Description2Code Dataset}},
    url = {https://github.com/ethancaballero/description2code},
    year = {2016}
}
"""

_HOMEPAGE = "https://github.com/ethancaballero/description2code"

_URL = "https://drive.google.com/uc?export=download&id=1gMtWvxBG6Oa7VCKekrk7nVGcd98oFp6-"

_NAMES = [
    "codechef_external",
    "codechef_hard",
    "codechef_harder",
    "codechef_hardest",
    "codechef_medium",
    "codechef_easy",
    "hackerearth_normal",
    "hackerearth_college",
    # "codeforces"
]


class Description2Code(datasets.GeneratorBasedBuilder):
    """Description2Code Dataset"""
    refactor = RefactoringTool(fixer_names=get_fixers_from_package("lib2to3.fixes"))

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
            version=version,
            description="Description2Code Dataset",
        )
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    def _generate_examples(self, basepath: str):
        name, level = self.config.name.split("_")

        if name == "codechef":
            problems_dir = os.path.join(basepath, name, level)
        elif name == "hackerearth":
            problems_dir = os.path.join(basepath, name, "problems_" + level)
        else:
            raise NotImplementedError

        for problem_dir in pathlib.Path(problems_dir).iterdir():
            if not problem_dir.is_dir():
                continue
            description_path = problem_dir.joinpath('description', 'description.txt')

            with open(description_path) as f:
                problem = f.read()

            solutions_dir = problem_dir.joinpath('solutions_python')

            if not solutions_dir.exists():
                continue

            for solution_path in solutions_dir.iterdir():
                if not solution_path.name.endswith('txt'):
                    continue
                solution_id = solution_path.name.split('.')[0]
                try:
                    with open(solution_path) as f:
                        solution = f.read()

                    solution += "\n"
                    solution = self.refactor.refactor_string(solution, str(problem_dir.name))
                except UnicodeDecodeError as e:
                    pdb.set_trace()
                    continue
                except Exception as e:
                    continue

                solution = reindent_code(str(solution))

                yield f'{problem_dir.name}-{solution_id}', {
                    'solution': solution,
                    'problem': problem,
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
        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "basepath": os.path.join(data_dir, 'description2code_current')
                }
            )
        ]
