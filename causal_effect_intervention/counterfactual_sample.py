import torch
from torchvision.models import vgg16
from src.lrp import LRPModel

def counterfactual_sample(x: torch.tensor, k: float, mode: str) -> torch.tensor:
    """

    Args:
        x: origin pic  b×3×H×W
        r: relation score b×H×W
        k: Top K percent significant /Top K percent not significant
        mode: causal saliency map or non causal saliency map


    Returns:causal saliency map

    """
    flag = x.size(1)
    if flag == 1:
        x = torch.squeeze(x)
        x = torch.stack([x, x, x], 1)  # b×3×H×W

    device = torch.device("cuda")
    model = vgg16(True)
    model.to(device)
    lrp_model = LRPModel(model)
    x = x.to(device)
    r = lrp_model.forward(x)

    r = r.to(device)
    one = torch.ones_like(torch.mean(r,0))
    zero = torch.zeros_like(torch.mean(r,0))

    tensor_list = []
    for chunk in torch.split(r, 1, dim=0): # Divide chunks according to batch size and find significant pixels in each chunk 1×H×W
        chunk = torch.squeeze(chunk) # H×W
        value_min, index_min = torch.kthvalue(chunk.view(-1), int(k * chunk.view(-1).numel()))
        value_max, index_max = torch.kthvalue(chunk.view(-1), int((1 - k) * chunk.view(-1).numel()))

        if mode == "non_causal":  # Randomly generate causal saliency maps, i.e. remove irrelevant pixels
            temp_proj = torch.where(chunk <= value_min, zero, one)  # Minimal mapping, sets the least significant k% of pixels in x to 0 1×H×W
        else:
            temp_proj = torch.where(chunk >= value_max, zero, one)  # Maximum mapping, sets the most significant k% of pixels in x to 0 1×H×W
        x_proj_i = torch.stack([temp_proj, temp_proj, temp_proj], 0)  # 3×H×W
        tensor_list.append(x_proj_i)

    x_proj = torch.stack(tensor_list, 0)  # b×3×H×W
    x_cf_temp = x.mul(x_proj)  # Hafdamard proj
    x_cf = torch.where(x_cf_temp == 0., one, x_cf_temp)  # mask, use white color: one

    if flag == 1:
        temp = torch.mean(x_cf, 1)
        x_cf = torch.unsqueeze(temp, 1)
        return x_cf
    else:
        return x_cf


