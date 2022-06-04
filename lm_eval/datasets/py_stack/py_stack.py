import os.path

import datasets
from datasets import DownloadManager, DatasetInfo
from lm_eval.code_utils import reindent_code

_CITATION = ""

_DESCRIPTION = ""

_HOMEPAGE = ""

_URLS = (
    "https://drive.google.com/uc?export=download&id=1c8BhOC4ydXIHiQyfpJhs5YBD_87UIueL"
)

_LICENSE = ""


class PyStack(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {"question": datasets.Value("string"), "code": datasets.Value("string")}
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        data_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "basepath": os.path.join(data_dir, "py-stack", "train"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "basepath": os.path.join(data_dir, "py-stack", "test"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "basepath": os.path.join(data_dir, "py-stack", "val"),
                },
            ),
        ]

    def _generate_examples(self, basepath):
        with (
            open(os.path.join(basepath, "code.txt"), mode="r") as code_f,
            open(os.path.join(basepath, "question.txt"), mode="r") as question_f,
        ):
            for i, (code, question) in enumerate(zip(code_f, question_f)):
                if len(question.strip()) == 0:
                    continue

                question = (
                    question.strip("\n")
                    .strip('"')
                    .strip("'")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                )
                code = (
                    code.strip("\n")
                    .strip('"')
                    .strip("'")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                )

                yield str(i), {"code": reindent_code(code), "question": question}
