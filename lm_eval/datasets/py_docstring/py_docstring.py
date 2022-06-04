import os.path

import datasets
from datasets import DownloadManager, DatasetInfo

from lm_eval.code_utils import extract_func_signature_and_body

_CITATION = ""

_DESCRIPTION = ""

_HOMEPAGE = ""

_URLS = (
    "https://drive.google.com/uc?export=download&id=1v8SMf1p2usKVnj1NHJsv_NjfLs13dNPO"
)

_LICENSE = ""

_NAMES = ["functions", "methods"]


def _replace_special_tokens(s: str) -> str:
    return (
        s.replace(" DCNL DCSP ", "\n\t").replace(" DCSP ", "\t").replace(" DCNL ", "\n")
    )


class PyDocstring(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=name, version=version, description=name)
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {
                "docstring": datasets.Value("string"),
                "declaration": datasets.Value("string"),
                "body": datasets.Value("string"),
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
                    "basepath": os.path.join(
                        data_dir, "py-docstring", self.config.name, "train"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "py-docstring", self.config.name, "test"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "py-docstring", self.config.name, "val"
                    ),
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
            open(
                os.path.join(basepath, "declaration.txt"),
                encoding="utf8",
                errors="ignore",
            ) as declaration_f,
        ):
            for i, (code, docstring, declaration) in enumerate(
                zip(
                    code_f.readlines(),
                    docstring_f.readlines(),
                    declaration_f.readlines(),
                )
            ):
                if len(docstring.strip()) == 0:
                    continue

                code = _replace_special_tokens(
                    code.strip("\n")
                    .strip('"')
                    .strip("'")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                )
                docstring = _replace_special_tokens(
                    docstring.strip("\n")
                    .strip('"')
                    .strip("'")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                )
                declaration = (
                    declaration.strip("\n")
                    .strip('"')
                    .strip("'")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                )

                func = declaration + "\n" + code

                try:
                    sig, body = extract_func_signature_and_body(func)
                except:
                    sig = declaration
                    body = code
                if sig is None or body is None:
                    continue

                yield str(i), {
                    "body": body,
                    "docstring": docstring,
                    "declaration": sig,
                }
