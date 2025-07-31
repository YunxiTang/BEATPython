from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn
import numpy as np
from PIL.Image import Image


def init_gpu(use_gpu=True, gpu_id=0):
    """initialize the device to use

    Args:
        use_gpu (bool, optional): prefer to use gpu is avaible. Defaults to True.

        gpu_id (int, optional): gpu id. Defaults to 0.

    Returns:
        device: the device to use
    """
    if torch.cuda.is_available() and use_gpu:
        print("Using GPU id {}".format(gpu_id))
        device = torch.device("cuda:" + str(gpu_id))
    else:
        print("GPU not detected. Defaulting to CPU.")
        device = torch.device("cpu")
    return device


def from_numpy(np_array, device=None):
    """
    put a variable to a tensor
    """
    if isinstance(np_array, torch.Tensor):
        return np_array.to(device)
    return torch.from_numpy(np_array).float().to(device)


def to_numpy(tensor):
    """
    convert a tensor to numpy variable
    """
    return tensor.to("cpu").detach().numpy()


def to_pil(img: torch.Tensor):
    img = np.moveaxis(img.numpy() * 255, 0, -1)
    return Image.fromarray(img.astype(np.uint8))


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def pad_remaining_dims(x, target):
    assert x.shape == target.shape[: len(x.shape)]
    return x.reshape(x.shape + (1,) * (len(target.shape) - len(x.shape)))


def dict_apply_split(
    x: Dict[str, torch.Tensor],
    split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
    x: List[Dict[str, torch.Tensor]],
    reduce_func: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def init_weights(modules):
    for module in modules:
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data, 1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data, 1.0)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
