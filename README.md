# mT0: Multilingual T0

`main.py` transfer the checkpoints from T5X (such as [this](https://console.cloud.google.com/storage/browser/bigscience-t5x/multilingual_t0/mt0_xl_t0pp/checkpoint_1025000;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)) to HF's PyTorch model.

First install [t5x](https://github.com/google-research/t5x/), [jax](https://github.com/google/jax) and [transformers](https://github.com/huggingface/transformers).

Then download the T5X checkpoints from Google Cloud bucket into the directory `t5x_ckpt_dir`. 

Run the following code to transfer weights from `t5x_ckpt_dir` T5X checkpoint to its corresponding model `hf_model` and save the model to `save_dir`. 
```
python3 main.py \
--hf_model "google/mt5-xl" \
--cache_dir "~/.cache/huggingface/transformers" \
--t5x_ckpt_dir "your/t5x_checkpoint_directory/" \
--save_dir "path/to/save_dir_of_transferred_checkpoint/"
```
