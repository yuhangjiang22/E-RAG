# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""ChemProt"""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

import math
from collections import defaultdict

label2word = {
    "CPR:9": "PRODUCT-OF",
    "CPR:4": "DOWNREGULATOR",
    "CPR:6": "ANTAGONIST",
    "CPR:5": "AGONIST",
    "CPR:3": "ACTIVATOR"
}


def get_all_triples(relations, text):
    triples = {}
    for relation in relations:
        for rel in relation:
            head = ' '.join(text[rel[0]:rel[1] + 1])
            tail = ' '.join(text[rel[2]:rel[3] + 1])
            label = rel[-1]
            triples[head + '-' + tail] = label2word[label]
    return triples

_DESCRIPTION = """\
ChemProt dataset.
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}


class ChemprotConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ChemprotConfig, self).__init__(**kwargs)


class Chemprot(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ChemprotConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "subject": datasets.Value("string"),
                    "object": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"],
                "dev": self.config.data_files["dev"],
                "test": self.config.data_files["test"],
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)

        f = list()
        with open(filepath, 'r') as file:
            for line in file:
                json_line = json.loads(line.strip())
                f.append(json_line)
        for id_, row in enumerate(f):
            text = [s for subs in row['sentences'] for s in subs]
            sentences = row['sentences']
            relations = row['relations']
            ners = row['ner']
            triples = get_all_triples(relations, text)
            key = 0
            start = 0
            for sentence, ner in zip(sentences, ners):
                curr_start = start
                start = curr_start + len(sentence)
                for subj in ner:
                    for obj in ner:
                        if subj[-1] == 'CHEMICAL' and obj[-1] == 'GENE':
                            head = ' '.join(text[subj[0]:subj[1] + 1])
                            tail = ' '.join(text[obj[0]:obj[1] + 1])
                            if head + '-' + tail not in triples:
                                label = 'No relationship'
                            else:
                                label = triples[head + '-' + tail]

                            t = []
                            for i, token in enumerate(sentence):
                                if subj[0] - curr_start == i:
                                    t.append('<subj>')
                                if subj[1] - curr_start + 1 == i:
                                    t.append('<subj/>')
                                if obj[0] - curr_start == i:
                                    t.append('<obj>')
                                if obj[1] - curr_start + 1 == i:
                                    t.append('<obj/>')
                                t.append(token)
                            t = ' '.join(t)
                            key += 1
                            yield str(row["doc_key"]) + '_{}'.format(str(key)), {
                                "subject": head,
                                "object": tail,
                                "title": str(row["doc_key"]) + '_{}'.format(str(key)),
                                "context": t,
                                "id": str(row["doc_key"]) + '_{}'.format(str(key)),
                                "label": label,
                            }