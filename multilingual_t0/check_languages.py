import datasets

MT0_LANG_TO_PROBS = {
    'en': 5.67,
    'ru': 3.71,
    'es': 3.09,
    'de': 3.05,
    'fr': 2.89,
    'it': 2.43,
    'pt': 2.36,
    'pl': 2.15,
    'nl': 1.98,
    'tr': 1.93,
    'ja': 1.92,
    'vi': 1.87,
    'id': 1.80,
    'cs': 1.72,
    'zh': 1.67,
    'fa': 1.67,
    'ar': 1.66,
    'sv': 1.61,
    'ro': 1.58,
    'el': 1.54,
    'uk': 1.51,
    'hu': 1.48,
    'da': 1.38,
    'fi': 1.35,
    'no': 1.33,
    'bg': 1.29,
    'hi': 1.21,
    'sk': 1.19,
    'ko': 1.14,
    'th': 1.14,
    'ca': 1.12,
    'ms': 1.09,
    'iw': 1.06,
    'lt': 1.04,
    'sl': 0.95,
    'mr': 0.93,
    'bn': 0.91,
    'et': 0.89,
    'lv': 0.87,
    'az': 0.82,
    'gl': 0.79,
    'cy': 0.76,
    'sq': 0.76,
    'ta': 0.73,
    'sr': 0.72,
    'ne': 0.69,
    'lb': 0.68,
    'hy': 0.65,
    'kk': 0.65,
    'ka': 0.64,
    'mt': 0.64,
    'af': 0.63,
    'fil': 0.62,
    'is': 0.62    
    }

dataset_list = list()
probs_list = list()
for lang, prob in MT0_LANG_TO_PROBS.items():
    print(lang)
    dataset_list.append(datasets.load_dataset("mc4", lang, split="train", streaming=True))
    probs_list.append(prob / 100)

print(sum(probs_list))