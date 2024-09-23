import torch
from config import *
from PIL import Image
from utils.utils import *
from model.load_model import load_model
from torchvision.transforms.functional import pil_to_tensor


# model selection
size = '7b' # [Select One] '0.5b' (transformers more recent version) | '1.8b' | '3.8b' (transformers==4.37.2) | '7b'

# User prompt
prompt_type="with_image" # Select one option "text_only", "with_image"
img_path='figures/demo.png'
question="Describe the image in detail"

# loading model
model, tokenizer = load_model(size=size)

# prompt type -> input prompt
if prompt_type == 'with_image':
    # Image Load
    image = pil_to_tensor(Image.open(img_path).convert("RGB"))
    inputs = [{'image': image, 'question': question}]
elif prompt_type=='text_only':
    inputs = [{'question': question}]

# cpu -> gpu
for param in model.parameters():
    if not param.is_cuda:
        param.data = param.cuda()

# Generate
with torch.inference_mode():

    # Model
    _inputs = model.eval_process(inputs=inputs,
                                data='demo',
                                tokenizer=tokenizer,
                                device='cuda:0')
    generate_ids = model.generate(**_inputs, do_sample=False, max_new_tokens=256)
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(answer)
