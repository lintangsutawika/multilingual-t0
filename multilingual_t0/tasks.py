import functools
import seqio
import tensorflow_datasets as tfds
import t5
from t5.evaluation import metrics
#from t5.data import preprocessors

import preprocessors

#vocabulary = seqio.SentencePieceVocabulary(
#    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
#output_features = {
#    'inputs': seqio.Feature(vocabulary=vocabulary),
#    'targets': seqio.Feature(vocabulary=vocabulary)
#}

MT5_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"

MT5_TEMPERATURE = 1.0 / 0.3
MT5_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=MT5_TEMPERATURE)

MT5_VOCAB = t5.data.SentencePieceVocabulary(MT5_SPM_PATH)
MT5_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=MT5_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=MT5_VOCAB, add_eos=True)
}

seqio.TaskRegistry.add(
    "mt5_test",
    source=seqio.TfdsDataSource(
        tfds_name="tydi_qa/goldp:2.1.0", splits=["train"]),
    preprocessors=[
        preprocessors.xquad,
        functools.partial(
            preprocessors.filter_tydiqa_by_language, lang="english"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=MT5_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])

#seqio.TaskRegistry.add(
#    'mt5_test',
#    #source=seqio.TfdsDataSource(tfds_name='wmt_t2t_translate/de-en:1.0.0'),
#    source=seqio.TfdsDataSource(tfds_name='m'),
#    preprocessors=[
#        functools.partial(
#            preprocessors.translate,
#            source_language='de', target_language='en'),
#        seqio.preprocessors.tokenize,
#        seqio.CacheDatasetPlaceholder(),
#        seqio.preprocessors.append_eos_after_trim,
#    ],
#    metric_fns=[metrics.bleu],
#    output_features=MT5_OUTPUT_FEATURES)
