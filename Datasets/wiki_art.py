import os
import pandas as pd

import datasets


_CITATION = """\
@InProceedings{huggingface:dataset,
title = {WikiArt},
author={Medellín AI.
},
year={2023}
}
"""

_DESCRIPTION = """\
Este dataset fue creado para el workshop de Medellin AI y Bancolombia con fines educativos.
"""

_HOMEPAGE = "https://www.meetup.com/medellin-ai/"

_LICENSE = "mit"

_URLS = {
    "train": "https://workshophuggingface.blob.core.windows.net/wikiart/train.zip",
    "test": "https://workshophuggingface.blob.core.windows.net/wikiart/test.zip"
}

_NAMES = ["Baroque", "Realism"]




class WikiArt(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")
    DEFAULT_WRITER_BATCH_SIZE = 200

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="All", version=VERSION, description="This contains the whole dataset"),
        datasets.BuilderConfig(name="Baroque", version=VERSION, description="This part of the dataset contains only Baroque style"),
        datasets.BuilderConfig(name="Realism", version=VERSION, description="This part of the dataset contains only Realism style"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "style": datasets.features.ClassLabel(names=_NAMES),
                "artwork": datasets.Value("string"),
                "image": datasets.Image(decode=True)
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=("image", "style"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        
        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "folderpath" : data_dir['train'],
                    "csv_file": 'wikiart_scraped_train.csv',
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "folderpath" : data_dir['test'],
                    "csv_file": 'wikiart_scraped_test.csv',
                    "split": "test"
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, folderpath, csv_file, split):


        df_wiki_art = pd.read_csv(os.path.join(folderpath,split,csv_file), header=0)

        if self.config.name != 'All':
            df_wiki_art.query(f"Style == '{self.config.name}'", inplace=True)

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        for index, row in df_wiki_art.iterrows():

            image_path = os.path.join(folderpath,split,row['Link'].split('/')[-1])
            # Yields examples as (key, example) tuples
            yield index, {
                "style": row["Style"],
                "artwork": row["Artwork"],
                "image": image_path
            }
