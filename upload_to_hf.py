from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="",
                    help='Huggingface Model name')
parser.add_argument('--hub_model_name', type=str, default="",
                    help='Huggingface Organization name')
parser.add_argument('--hf_org', type=str, default="",
                    help='Huggingface Organization name')
parser.add_argument('--use_temp_dir', type=bool, default=False,
                    help='Use this option if you see "you need to pass Repository a valid git clone" error')

args = parser.parse_args()


model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
print("✅ Loaded model")

# For some reason, push_to_hub fails if use_temp_dir is False
tokenizer.push_to_hub(args.hub_model_name, organization=args.hf_org, use_temp_dir=args.use_temp_dir)
model.push_to_hub(args.hub_model_name, organization=args.hf_org, use_temp_dir=args.use_temp_dir)
print("✅ Success")
