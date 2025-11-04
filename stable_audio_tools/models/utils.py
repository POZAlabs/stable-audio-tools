import os

import torch
from safetensors.torch import load_file
from torch.nn.utils import remove_weight_norm


def copy_state_dict(model, state_dict):
    """
    Load a source state_dict (with 'weight' only) into a model that uses weight_norm ('weight_g', 'weight_v').
    It decomposes 'weight' from the source into 'g' and 'v' for the target model.
    """
    target_state_dict = model.state_dict()
    for key, value in state_dict.items():
        if key.endswith(".weight"):
            base_name = key.rsplit(".weight", 1)[0]
            g_key = base_name + ".weight_g"
            v_key = base_name + ".weight_v"

            # 타겟 모델에 g와 v 파라미터가 모두 있는지 확인 (weight_norm 적용 여부)
            if g_key in target_state_dict and v_key in target_state_dict:
                weight = state_dict[key]
                norm_dims = tuple(range(1, weight.dim()))
                norm = torch.norm(weight, p=2, dim=norm_dims, keepdim=True)

                if target_state_dict[v_key].shape == weight.shape and target_state_dict[g_key].shape == norm.shape:
                    target_state_dict[g_key].copy_(norm)
                    target_state_dict[v_key].copy_(weight)
                else:
                    print(f"  [!] Shape mismatch for '{key}'. Skipping.")
                    print(f"      - Target g shape: {target_state_dict[g_key].shape}, Calculated norm shape: {norm.shape}")
                    print(f"      - Target v shape: {target_state_dict[v_key].shape}, Source weight shape: {weight.shape}")

                continue

        # Case 2: 일반 파라미터 (bias 등) 또는 weight_norm이 아닌 weight
        if key in target_state_dict and target_state_dict[key].shape == value.shape:
            target_state_dict[key].copy_(value)
        else:
            print(f"Warning: Key '{key}' not found in model or shape mismatch. Skipping.")

    # 3. 최종적으로 채워진 state_dict를 모델에 로드
    model.load_state_dict(target_state_dict, strict=True)
    print("\nState dict loaded successfully with weight_norm conversion.")


def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False).state_dict()

    return state_dict


def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)

    return model


try:
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True
except Exception as e:
    pass

# Get torch.compile flag from environment variable ENABLE_TORCH_COMPILE

enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"


def compile(function, *args, **kwargs):

    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function

    return function

# Sampling functions copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/utils.py under MIT license
# License can be found in LICENSES/LICENSE_META.txt


def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def next_power_of_two(n):
    return 2 ** (n - 1).bit_length()


def next_multiple_of_64(n):
    return ((n + 63) // 64) * 64
