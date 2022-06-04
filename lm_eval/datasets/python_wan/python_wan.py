import os.path

import datasets
from datasets import DownloadManager, DatasetInfo

from lm_eval.code_utils import extract_func_signature_and_body

_CITATION = ""

_DESCRIPTION = ""

_HOMEPAGE = ""

_URLS = (
    "https://drive.google.com/uc?export=download&id=14CZdej16ey7Z5PFN8uBGCb4uNEMUXCxu"
)

_LICENSE = ""


class PythonWan(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

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
        data_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "basepath": os.path.join(data_dir, "python_wan", "train"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "basepath": os.path.join(data_dir, "python_wan", "test"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "basepath": os.path.join(data_dir, "python_wan", "val"),
                },
            ),
        ]

    def _generate_examples(self, basepath):
        with (
            open(
                os.path.join(basepath, "code.txt"), encoding="utf8", errors="ignore"
            ) as code_f,
            open(
                os.path.join(basepath, "docstring.txt"),
                encoding="utf8",
                errors="ignore",
            ) as docstring_f,
        ):
            for i, (code, docstring) in enumerate(zip(code_f, docstring_f)):
                if len(docstring.strip()) == 0:
                    continue

                docstring = (
                    docstring.strip("\n")
                    .strip('"')
                    .strip("'")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                )

                try:
                    sig, body = extract_func_signature_and_body(
                        code.strip("\n")
                        .strip('"')
                        .strip("'")
                        .replace("\\n", "\n")
                        .replace("\\t", "\t")
                    )
                except:
                    body = code

                yield str(i), {
                    "body": body,
                    "docstring": docstring,
                    "declaration": sig,
                }
