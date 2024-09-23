from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
from torch import nn
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_phantom import PhantomConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_internlm2 import InternLM2ForCausalLM

from utils.utils import *

class PhantomForCausalLM(PreTrainedModel):
    config_class = PhantomConfig
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'InternLM2DecoderLayer']

    def __init__(self, config: PhantomConfig):
        super().__init__(config)
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio

        self.vision_model = InternVisionModel(config.vision_config)
        self.language_model = InternLM2ForCausalLM(config.llm_config)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # prompt rule
        self.prompt_rule = {
                            "system_start": "<|im_start|>system\n",
                            "system_end": "<|im_end|>",
                            "user_start": "<|im_start|>user\n",
                            "user_end": "<|im_end|>",
                            "assistant_start": "<|im_start|>assistant\n",
                            "assistant_end": "<|im_end|>",
                            "test_start": "assistant\n",
                            "test_end": "<|im_end|>",
                            "split": "",
                            }

    def eval_process(
        self,
        inputs,
        tokenizer,
        data,
        device,
    ):
        batched_image=[]
        batched_qa_prompt=[]
        batched_phantom_position = []
        for _input in inputs:

            # making image prompt
            if 'image' in _input.keys() and _input['image'] != None:
                process_image = dynamic_preprocess(_input['image'].to(device))
                dynamic_process_image = torch.stack([dynamic_transform(image) for image in process_image]).to(device)
                img_token_number = dynamic_process_image.shape[0] * 256
                batched_image.append(dynamic_process_image)

            # make question and answer
            question =  _input['question']

            # make instruction (qa pair) and label 
            qa_prompt = make_instruction(question, data, self.prompt_rule)
            
            # adding image special tokens to question
            if 'image' in _input.keys():
                qa_prompt = qa_prompt.replace('<image>', '<img><IMG_CONTEXT></img>')

                # add bundle image tokens if it has <image> token
                qa_prompt = add_bundle_tokens(qa_prompt, '<IMG_CONTEXT>', img_token_number) 

            # phantom_position
            label = tokenizer(qa_prompt, return_tensors='pt', add_special_tokens=False).input_ids[0].to(device)
            phantom_position = torch.zeros_like(label)
            phantom_position[0] = 1

            # batched processing
            batched_qa_prompt.append(qa_prompt)
            batched_phantom_position.append(phantom_position.flip(dims=[0]))

        '''For Final Outputs'''
        qa_prompts = tokenizer(batched_qa_prompt, padding='longest', return_tensors="pt", add_special_tokens=False)

        # [1] input_ids
        input_ids = qa_prompts.input_ids.to(device)
  
        # [2] attention_mask
        attention_mask = qa_prompts.attention_mask.to(device)

        # [3] Phantom Position
        batched_phantom_position = torch.nn.utils.rnn.pad_sequence(batched_phantom_position, batch_first=True, padding_value=0).flip(dims=[1]) # padding left

        if len(batched_image):
            return {"input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "pixel_values": torch.cat(batched_image, dim=0).to(device),
                    "phantom_position": batched_phantom_position.bool()
                    }
        else:
            return {"input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "phantom_position": batched_phantom_position.bool()
                    }

    def extract_feature(self, pixel_values):
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.vision_proj(vit_embeds)
        return vit_embeds

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            phantom_position: torch.BoolTensor = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values.to(torch.bfloat16))
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.config.image_token_index)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            phantom_position=phantom_position,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            pad_token_id=self.config.eos_token_id,
            eos_token_id=self.config.eos_token_id,
            **generate_kwargs,
        )

        return outputs
