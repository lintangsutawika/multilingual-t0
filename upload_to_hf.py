from transformers import AutoModel, AutoModelForSeq2SeqLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="",
                    help='Huggingface Model name')
parser.add_argument('--hub_model_name', type=str, default="",
                    help='Huggingface Organization name')
parser.add_argument('--hf_org', type=str, default="",
                    help='Huggingface Organization name')

args = parser.parse_args()


model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
print("✅ Loaded model")
model.push_to_hub(args.hub_model_name, organization=args.hf_org)
print("✅ Success")