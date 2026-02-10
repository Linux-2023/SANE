import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['HF_HOME'] = "/workspace/pjk/ELM/openpi/huggingface"
from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "<image> What is the animals in the picture?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, text=prompt,  return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs,)
answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Answer:", answer)