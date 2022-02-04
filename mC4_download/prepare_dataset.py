import os
import datasets

_DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/multilingual/c4-{language}{split_suffix}.tfrecord-{index:05d}-of-{n_shards:05d}.json.gz"

_LANGUAGES = [
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bg-Latn",
    "bn",
    "ca",
    "ceb",
    "co",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "el-Latn",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "haw",
    "hi",
    "hi-Latn",
    "hmn",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "iw",
    "ja",
    "ja-Latn",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "ny",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "ru-Latn",
    "sd",
    "si",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "st",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "und",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zh-Latn",
    "zu",
]

_N_SHARDS_PER_SPLIT = {
    "af": {"train": 64, "validation": 1},
    "am": {"train": 16, "validation": 1},
    "ar": {"train": 1024, "validation": 4},
    "az": {"train": 256, "validation": 1},
    "be": {"train": 128, "validation": 1},
    "bg": {"train": 1024, "validation": 1},
    "bg-Latn": {"train": 4, "validation": 1},
    "bn": {"train": 512, "validation": 1},
    "ca": {"train": 512, "validation": 1},
    "ceb": {"train": 8, "validation": 1},
    "co": {"train": 8, "validation": 1},
    "cs": {"train": 1024, "validation": 2},
    "cy": {"train": 256, "validation": 1},
    "da": {"train": 1024, "validation": 1},
    "de": {"train": 2048, "validation": 16},
    "el": {"train": 1024, "validation": 2},
    "el-Latn": {"train": 16, "validation": 1},
    "en": {"train": 11264, "validation": 128},
    "eo": {"train": 32, "validation": 1},
    "es": {"train": 2048, "validation": 16},
    "et": {"train": 256, "validation": 1},
    "eu": {"train": 64, "validation": 1},
    "fa": {"train": 1024, "validation": 2},
    "fi": {"train": 1024, "validation": 1},
    "fil": {"train": 64, "validation": 1},
    "fr": {"train": 2048, "validation": 16},
    "fy": {"train": 16, "validation": 1},
    "ga": {"train": 16, "validation": 1},
    "gd": {"train": 16, "validation": 1},
    "gl": {"train": 128, "validation": 1},
    "gu": {"train": 64, "validation": 1},
    "ha": {"train": 8, "validation": 1},
    "haw": {"train": 2, "validation": 1},
    "hi": {"train": 1024, "validation": 2},
    "hi-Latn": {"train": 16, "validation": 1},
    "hmn": {"train": 8, "validation": 1},
    "ht": {"train": 8, "validation": 1},
    "hu": {"train": 1024, "validation": 2},
    "hy": {"train": 128, "validation": 1},
    "id": {"train": 1024, "validation": 4},
    "ig": {"train": 4, "validation": 1},
    "is": {"train": 128, "validation": 1},
    "it": {"train": 1024, "validation": 8},
    "iw": {"train": 1024, "validation": 1},
    "ja": {"train": 1024, "validation": 8},
    "ja-Latn": {"train": 8, "validation": 1},
    "jv": {"train": 8, "validation": 1},
    "ka": {"train": 256, "validation": 1},
    "kk": {"train": 256, "validation": 1},
    "km": {"train": 64, "validation": 1},
    "kn": {"train": 64, "validation": 1},
    "ko": {"train": 1024, "validation": 1},
    "ku": {"train": 16, "validation": 1},
    "ky": {"train": 64, "validation": 1},
    "la": {"train": 64, "validation": 1},
    "lb": {"train": 32, "validation": 1},
    "lo": {"train": 8, "validation": 1},
    "lt": {"train": 512, "validation": 1},
    "lv": {"train": 256, "validation": 1},
    "mg": {"train": 8, "validation": 1},
    "mi": {"train": 4, "validation": 1},
    "mk": {"train": 128, "validation": 1},
    "ml": {"train": 128, "validation": 1},
    "mn": {"train": 128, "validation": 1},
    "mr": {"train": 1024, "validation": 1},
    "ms": {"train": 512, "validation": 1},
    "mt": {"train": 128, "validation": 1},
    "my": {"train": 64, "validation": 1},
    "ne": {"train": 256, "validation": 1},
    "nl": {"train": 1024, "validation": 4},
    "no": {"train": 1024, "validation": 1},
    "ny": {"train": 4, "validation": 1},
    "pa": {"train": 32, "validation": 1},
    "pl": {"train": 1024, "validation": 4},
    "ps": {"train": 16, "validation": 1},
    "pt": {"train": 1024, "validation": 4},
    "ro": {"train": 1024, "validation": 2},
    "ru": {"train": 4096, "validation": 32},
    "ru-Latn": {"train": 32, "validation": 1},
    "sd": {"train": 64, "validation": 1},
    "si": {"train": 64, "validation": 1},
    "sk": {"train": 512, "validation": 1},
    "sl": {"train": 256, "validation": 1},
    "sm": {"train": 4, "validation": 1},
    "sn": {"train": 8, "validation": 1},
    "so": {"train": 64, "validation": 1},
    "sq": {"train": 128, "validation": 1},
    "sr": {"train": 256, "validation": 1},
    "st": {"train": 2, "validation": 1},
    "su": {"train": 4, "validation": 1},
    "sv": {"train": 1024, "validation": 2},
    "sw": {"train": 32, "validation": 1},
    "ta": {"train": 256, "validation": 1},
    "te": {"train": 128, "validation": 1},
    "tg": {"train": 64, "validation": 1},
    "th": {"train": 1024, "validation": 1},
    "tr": {"train": 1024, "validation": 4},
    "uk": {"train": 1024, "validation": 2},
    "und": {"train": 3072, "validation": 32},
    "ur": {"train": 128, "validation": 1},
    "uz": {"train": 32, "validation": 1},
    "vi": {"train": 1024, "validation": 4},
    "xh": {"train": 2, "validation": 1},
    "yi": {"train": 16, "validation": 1},
    "yo": {"train": 2, "validation": 1},
    "zh": {"train": 1024, "validation": 2},
    "zh-Latn": {"train": 8, "validation": 1},
    "zu": {"train": 8, "validation": 1},
}

download_fraction = {
    "en": 0.2,
    "ru": 0.5,
    "es": 0.8,
    "de": 0.8,
    "fr": 0.9,    
}

def download_mc4(languages=["id", "zu", "ja"], base_dir = "./", num_process=8):

    mc4_datasets = {}
    for lang in languages:

        n_shards = _N_SHARDS_PER_SPLIT[lang]['train']
        
        if lang in download_fraction:
            n_shards = int(n_shards*download_fraction[lang])+1 # round up

        data_urls = {
            'train': [
                _DATA_URL.format(
                    language=lang,
                    split_suffix='',
                    index=idx,
                    n_shards=n_shards) for idx in range(n_shards)
            ]
        }

        cache_dir = os.path.join(base_dir, lang)
        download_config = datasets.utils.DownloadConfig(
                cache_dir=cache_dir,
                num_proc=num_process
                )
        dl_manager = datasets.DownloadManager(
                download_config=download_config
                )

        train_dataset_files = dl_manager.download(data_urls["train"])
        mc4_datasets[lang] = {
            "cache_dir": cache_dir,
            }

    return mc4_datasets


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Arguments for download mC4')
    parser.add_argument('--base_dir', type=str, help="Where to store the mC4 subsets")
    parser.add_argument('--num_process', type=int, help="How many cores to process")
    args = parser.parse_args()

    print(download_mc4(
        languages=_LANGUAGES,
        base_dir=args.base_dir,
        num_process=args.num_process
    ))