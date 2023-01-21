__all__ = [
    "build_bound_str",
    "log_certification_ratio",
]

import logging
from typing import NoReturn, Optional, Union

import torch
from torch import BoolTensor, LongTensor, Tensor

from .. import _config as config
from .. import learner_ensemble

PLOT_QUANTILE = 0.05
DEFAULT_N_BINS = 20


def log_certification_ratio(model: learner_ensemble.DisjointEnsemble, bound: LongTensor,
                            k: Optional[int] = None) -> NoReturn:
    r""" Log the certification ratio for the predictions """
    if len(bound.shape) > 1:
        bound = bound.squeeze(dim=1)
    assert len(bound.shape) == 1, "Unexpected size of the bound tensors"

    header = f"{model.name()}"

    for base_bound in [0]:
        _print_cert_res(header=header, model=model, bound_cnt=base_bound,
                        bound_mask=bound >= base_bound, k=k)

    for i in range(1, bound.max().item() + 1):
        bound_mask = bound >= i
        _print_cert_res(header=header, model=model, bound_cnt=i, bound_mask=bound_mask, k=k)


def _print_cert_res(header: str, k: int, model: learner_ensemble.DisjointEnsemble,
                    bound_cnt: int, bound_mask: BoolTensor) -> float:
    r"""
    Standardizes printing the certification results
    :return: Prints the certified ratio
    """
    tot_count = bound_mask.numel()

    # Optionally log the top-k number
    mid_str = "Cert."
    if k is not None and k > 1:
        mid_str += f" (Top k={k})"

    cert_count = torch.sum(bound_mask).item()
    logging.info(f"{header} {mid_str} Model # Submodels: {model.n_models}")
    if config.is_ssl():
        logging.info(f"{header} {mid_str} Model SSL Degree: {config.SSL_DEGREE}")
    logging.info(f"{header} {mid_str} Bound: {bound_cnt}")

    ratio = cert_count / tot_count
    logging.info(f"{header} {mid_str} Count: {cert_count} / {tot_count} ({ratio:.2%})")
    return ratio


def build_bound_str(bound_dist: Union[float, int, str]) -> str:
    r""" Construct the bound string from the distance """
    assert isinstance(bound_dist, (float, int, str)), f"Type is {bound_dist.__class__.__name__}"
    bound_str = str(bound_dist)
    if config.IS_BOUND_PERCENT:
        bound_str += "%"
    return bound_str


def calc_bound_val(dist: float, y: Tensor) -> Union[float, Tensor]:
    r""" Standardizes the calculation of the bound value """
    if config.IS_BOUND_PERCENT:
        return dist / 100 * y
    return torch.full_like(y, dist)
