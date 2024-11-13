# Licensed under the MIT license.

import torch
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import os

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True
)

def load_HF_model(ckpt) -> tuple:
    with open(f"{os.environ.get('PWD')}/models/token.txt", "r") as f:
       token = f.read().rstrip()
    tokenizer = AutoTokenizer.from_pretrained(ckpt, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        token=token
    )
    return tokenizer, model


def generate_with_HF_model(
    tokenizer, model, input=None, temperature=0.8, top_p=0.95, top_k=40, num_beams=1, max_new_tokens=128, **kwargs
):
    #try:
    inputs = tokenizer(input, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=40,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    #except Exception as e:
        #breakpoint()
    #    print("Exception:", e)
    return output
