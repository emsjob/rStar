# Licensed under the MIT license.

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math
import torch
import torch.nn.functional as F

def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=256):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
        )

    return tokenizer, llm

def calculate_entropy(logits):
    logp = F.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    entropy = -torch.sum(p * logp, dim=-1)
    return entropy.item()

def adaptive_temperature(logits, entropy):
    polyfit = np.array([-0.037, 0.481, -2.3, 4.917, -1.791])
    beta = np.where(entropy > 0.5, np.max([np.polyval(polyfit, entropy), 1.0]), 1.0)
    return beta.item() * logits

def logit_proc(prev_tokens, logits):
    entropy = calculate_entropy(logits)
    updated_logits = adaptive_temperature(logits, entropy)
    return updated_logit

def generate_with_vLLM_model(
    model,
    input,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        logits_processors=[logit_proc],
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    return output


if __name__ == "__main__":
    model_ckpt = "mistralai/Mistral-7B-v0.1"
    tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False)
    input = "What is the meaning of life?"
    output = generate_with_vLLM_model(model, input)
    breakpoint()
    print(output[0].outputs[0].text)
