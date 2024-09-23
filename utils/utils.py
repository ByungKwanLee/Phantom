import gc
import math
import torch
from config import *
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import pil_to_tensor

output_filtering = lambda x, model: x.split(model.prompt_rule["test_start"])[-1].split(model.prompt_rule["test_end"])[0].strip()
def memory_optimization():
    # memory deallocation
    gc.collect()

    # removing cache
    torch.cuda.empty_cache()

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad=False

def find_special_token(string, special_token):
    start = 0
    while True:
        start = string.find(special_token, start)
        if start == -1: return
        yield start
        start += len(special_token) # use start += 1 to find overlapping matches

def add_bundle_tokens(input_string, special_token, num):

    # number of special tokens in input_string
    num_special_tokens = len(list(find_special_token(input_string, special_token)))

    # No special token -> return the raw
    if not num_special_tokens:
        return input_string
    
    result = ""
    index = 0
    while index < len(input_string):
        if input_string[index:index + len(special_token)] == special_token:
            result += special_token * num
            index += len(special_token)
        else:
            result += input_string[index]
            index += 1

    assert len(list(find_special_token(result, special_token))) == num_special_tokens * num
    return result

def make_instruction_and_label(question, answer, tokenizer, device, prompt_rule, config):

    qa_prompt = make_human_string(prompt_rule["user_start"]+question+prompt_rule["user_end"],
                                  prompt_rule["assistant_start"],
                                  split=prompt_rule["split"])
    
    # Only QA Prompt Length
    length = tokenizer(qa_prompt, return_tensors='pt', add_special_tokens=False).input_ids[0].shape[0]

    # Concat QA Prompt + Answer Length + stop token
    qa_prompt = qa_prompt + answer + prompt_rule["assistant_end"]

    # label
    label = tokenizer(qa_prompt, return_tensors='pt', add_special_tokens=False).input_ids[0].to(device)

    # phantom_position
    phantom_position = torch.zeros_like(label)
    phantom_position[0] = 1

    # add ignore index to label
    label[:length] = config.ignore_index

    return qa_prompt, label, phantom_position

def make_instruction(question, dataset, prompt_rule):
    
    if dataset != "mathverse" and dataset != "hallusionbench" and dataset == "demo":
        question = "<image>" + question

    if dataset in ["sqa", "mmbench", "mmbench_cn", "mmbench_dev", "mmbench_cn_dev", "seed", "seed-2-plus", "qbench", "ai2d", "mmstar", "cvbench", "blink"]:
        question = question + "\nAnswer with the option's letter from the given choices directly."

    elif dataset in ["pope", "chartqa"]:
        question = question + "\nAnswer the question using a single word or phrase."
        
    elif dataset in ["hallusionbench"]:
        if "Please answer yes or no." not in question:
            question = question + "\nPlease answer yes or no."
    
    qa_prompt = make_human_string(prompt_rule["user_start"]+question+prompt_rule["user_end"],
                                  prompt_rule["assistant_start"],
                                  split=prompt_rule["split"])

    return qa_prompt

def make_human_string(*args, split):
    out = ''
    for i, arg in enumerate(args):
        out += arg
        if i != len(args)-1:
            out += split
    return out

def get_max_new_tokens(data_name):
    if data_name.lower() in ["mme", "pope", "sqa", "mmbench", "mmbench_cn", \
                             "mmbench_dev","mmbench_cn_dev", "seed", "seed-2-plus", \
                             "qbench", "ai2d", "mmstar", "chartqa", "hallusionbench", \
                             "cvbench", "blink"]:
        return 5
    elif data_name.lower() in ["llava", "llava_wilder", "mm-vet", "mm-vet-v2"]:
        return 1024
    elif data_name.lower() in ["mathvista", "mathverse", "visualwebbench"]:
        return 512
    else:
        raise Exception("Check Data Name!")

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

class XAttention(nn.Module):

    def __init__(self,
                 in_features,
                 activation=F.gelu,
                 eta=1e-4):
        """XAttention attention.
        :param in_features: Size of each input sample.
        :param activation: The activation after each linear transformation.
        """
        super(XAttention, self).__init__()
        self.in_features = in_features
        self.activation = activation
        self.linear_q = nn.Linear(in_features, in_features, False)
        self.linear_k = nn.Linear(in_features, in_features, False)
        self.linear_v = nn.Linear(in_features, in_features, False)
        self.linear_o = nn.Linear(in_features, in_features, False)
        self.eta = eta

    def forward(self, q, k, v, is_residual=False):
        _q, _k, _v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            _q = self.activation(_q)
            _k = self.activation(_k)
            _v = self.activation(_v)
        y = ScaledDotProductAttention()(_q, _k, _v)
        y = self.linear_o(y)
        if self.activation is not None: y = self.activation(y)
        return q + self.eta*y if is_residual else self.eta*y

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
dynamic_transform = build_transform(input_size=448)

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    image = to_pil_image(image)
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def concat_images_horizontally_with_margin(image_tensors, margin=10):
    images = [to_pil_image(xx) for xx in image_tensors]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    
    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one
    return pil_to_tensor(new_image)