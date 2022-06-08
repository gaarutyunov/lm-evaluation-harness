import os.path

import datasets
import jsonlines
from datasets import DownloadManager, DatasetInfo

_DESCRIPTION = """\
CMU CoNaLa is a dataset crawled from Stack Overflow, automatically filtered, then curated by 
annotators, split into 2,379 training and 500 test examples (read more about the process here). It provides a 
large automatically-mined dataset with 600k examples. These data sets can be 
used for the CoNaLa challenge, or for any other research on the intersection of code and natural language.
"""

_LICENSE = ""

_HOMEPAGE = "https://conala-corpus.github.io/"

_CITATION = """\
@inproceedings{yin2018mining,
  author = {Yin, Pengcheng and Deng, Bowen and Chen, Edgar and Vasilescu, Bogdan and Neubig, Graham},
  title = {Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow},
  booktitle = {International Conference on Mining Software Repositories},
  series = {MSR},
  pages = {476--486},
  year = {2018},
  publisher = {ACM},
  doi = {https://doi.org/10.1145/3196398.3196408},
}
"""

_URL = "https://www.phontron.com/download/conala-corpus-v1.1.zip"


class CoNaLa(datasets.GeneratorBasedBuilder):
    def _generate_examples(self, basepath):
        with jsonlines.open(os.path.join(basepath, 'conala-corpus', 'conala-mined.jsonl')) as reader:
            for line in reader.iter():
                yield line['question_id'], {
                    'question': line['intent'],
                    'answer': line['snippet']
                }

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
        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'basepath': data_dir
                },
            ),
        ]