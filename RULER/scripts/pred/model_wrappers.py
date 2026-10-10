# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional

###
from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral
from evaluation.qwen import enable_tuple_kv_cache_for_qwen2

logger = logging.getLogger(__name__)


class HuggingFaceModel:
    def __init__(self, name_or_path: str, args, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        # ============== Random Seed 설정 (for reproducibility) ==============
        import numpy as np
        import random

        seed = getattr(args, 'random_seed', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # ====================================================================

        # ************************************************************************
        if 'llama' in name_or_path.lower() or 'longchat' in name_or_path.lower() or 'vicuna' in name_or_path.lower():
            enable_tuple_kv_cache_for_llama()
        if 'mistral' in name_or_path.lower():
            enable_tuple_kv_cache_for_mistral()
        if 'qwen' in name_or_path.lower():
            enable_tuple_kv_cache_for_qwen2()
        # ************************************************************************
        
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        # ************************************************************************
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path, 
            device_map="auto", 
            trust_remote_code=True, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        
        # ****************************************************************************************************************
        from evaluation.quest_attention import enable_quest_attention_eval

        enable_quest_attention_eval(self.model, self.args)
        self.pipeline=None
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     trust_remote_code=True,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     model_kwargs=model_kwargs,
        # )
        # ************************************************************************

        self.generation_kwargs = generation_kwargs
        """
        do_sample=args.temperature > 0,
        repetition_penalty=1,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop=args.stop_words,
        max_new_tokens=tokens_to_generate,
        """
        
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]
    
    def process_batch(
        self,
        prompts: List[str],
        # 🔫 ***************************************
        task_sample_idx: Optional[List[int]],
        # 🔫 ***************************************
        **kwargs
    ) -> List[dict]:
        with torch.inference_mode():
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            # ************ ************ ************ ************ ************
            output = self.model(
                input_ids=inputs.input_ids,
                # ***********************************************
                attention_mask=inputs.attention_mask,  # 이 줄 추가
                # ***********************************************
                past_key_values=None,
                use_cache=True,
                # ************************************************
                task_sample_idx=task_sample_idx
                # ************************************************
                # **self.generation_kwargs
            )
            past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]

            max_new_tokens = self.generation_kwargs['max_new_tokens']
            for _ in range(max_new_tokens-1):
                outputs = self.model( # old_mistral_for_causal_lm_forward
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                    # ************************************************
                    task_sample_idx=task_sample_idx
                    # ************************************************
                    # **self.generation_kwargs
                )
                
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]

                if pred_token_idx.item() == self.tokenizer.eos_token_id:
                    break
            # ************ ************ ************ ************ ************
        
        generated_texts = self.tokenizer.batch_decode([generated_content], skip_special_tokens=True)
        # print(f"--------------------\n✅generated_texts: {generated_texts}\n---------------------------")
        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})
        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
