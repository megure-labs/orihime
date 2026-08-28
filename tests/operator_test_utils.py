# SPDX-License-Identifier: Apache-2.0
"""Test helpers that compose the public named low-level primitives.

The numerical suites compare several kernel levels side by side. These helpers
only assemble values that are separate in the uniform API; they do not add
runtime behavior to the package.
"""

from __future__ import annotations

try:
    from orihime import ops as _public_ops

    cky = _public_ops._kernels["cky"]
    damerau = _public_ops._kernels["damerau"]
    dtw = _public_ops._kernels["dtw"]
    eisner = _public_ops._kernels["eisner"]
    lcs = _public_ops._kernels["lcs"]
    lev = _public_ops._kernels["lev"]
    mas = _public_ops._kernels["mas"]
    nw = _public_ops._kernels["nw"]
    nw_affine = _public_ops._kernels["nw_affine"]
    osa = _public_ops._kernels["osa"]
    sv_affine = _public_ops._kernels["sv_affine"]
    sv_linear = _public_ops._kernels["sv"]
    sw = _public_ops._kernels["sw"]
    sw_affine = _public_ops._kernels["sw_affine"]
except ImportError:
    pass


def _field(index, functions, message, *args):
    if index < 0 or index >= len(functions):
        raise RuntimeError(message)
    return functions[index](*args)


def sw_forward_with_grads(scores, gap, temp, lengths):
    value, marginals = sw.forward(scores, gap, temp, lengths)
    grad_gap, grad_temp = sw.value_grad_params(
        scores, gap, temp, lengths
    )
    return value, marginals, grad_gap, grad_temp


def sw_param_field(scores, index, gap, temp, lengths):
    return _field(
        index,
        (sw.marginals_grad_gap, sw.marginals_grad_temp),
        "param_type must be 0 or 1",
        scores,
        gap,
        temp,
        lengths,
    )


def sw_affine_forward_with_grads(
    scores, gap_open, gap_ext, temp, lengths
):
    value, marginals = sw_affine.forward(
        scores, gap_open, gap_ext, temp, lengths
    )
    grad_open, grad_ext, grad_temp = sw_affine.value_grad_params(
        scores, gap_open, gap_ext, temp, lengths
    )
    return value, marginals, grad_open, grad_ext, grad_temp


def sw_affine_param_field(
    scores, index, gap_open, gap_ext, temp, lengths
):
    return _field(
        index,
        (
            sw_affine.marginals_grad_gap_open,
            sw_affine.marginals_grad_gap_ext,
            sw_affine.marginals_grad_temp,
        ),
        "param_type must be 0, 1, or 2",
        scores,
        gap_open,
        gap_ext,
        temp,
        lengths,
    )


def nw_forward_with_grads(scores, gap, temp, lengths):
    value, marginals = nw.forward(scores, gap, temp, lengths)
    grad_gap, grad_temp = nw.value_grad_params(
        scores, gap, temp, lengths
    )
    return value, marginals, grad_gap, grad_temp


def nw_param_field(scores, index, gap, temp, lengths):
    return _field(
        index,
        (nw.marginals_grad_gap, nw.marginals_grad_temp),
        "param_type must be 0 or 1",
        scores,
        gap,
        temp,
        lengths,
    )


def nw_affine_forward_with_grads(
    scores, gap_open, gap_ext, temp, lengths
):
    value, marginals = nw_affine.forward(
        scores, gap_open, gap_ext, temp, lengths
    )
    _, grad_open, grad_ext, grad_temp = nw_affine.value_grad_params(
        scores, gap_open, gap_ext, temp, lengths
    )
    return value, marginals, grad_open, grad_ext, grad_temp


def nw_affine_param_field(
    scores, index, gap_open, gap_ext, temp, lengths
):
    return _field(
        index,
        (
            nw_affine.marginals_grad_gap_open,
            nw_affine.marginals_grad_gap_ext,
            nw_affine.marginals_grad_temp,
        ),
        "param_type must be 0, 1, or 2",
        scores,
        gap_open,
        gap_ext,
        temp,
        lengths,
    )


def dtw_forward_with_grads(costs, temp, lengths, bandwidth):
    value, marginals = dtw.forward(costs, temp, lengths, bandwidth)
    grad_temp = dtw.value_grad_params(
        costs, temp, lengths, bandwidth
    )
    return value, marginals, grad_temp


def cky_forward_with_grads(merge_scores, leaf_scores, temp):
    value, marginals = cky.forward(merge_scores, leaf_scores, temp)
    grad_leaf, grad_temp = cky.value_grad_params(
        merge_scores, leaf_scores, temp
    )
    return value, marginals, grad_leaf, grad_temp


def mas_forward_with_grads(scores, temp, lengths):
    return mas.forward(scores, temp, lengths)


def mas_full_outputs(scores, temp, lengths):
    value, marginals = mas.forward(scores, temp, lengths)
    grad_temp = mas.value_grad_params(scores, temp, lengths)
    return value, marginals, grad_temp


def eisner_forward_with_grads(arc_scores, temp, lengths):
    return eisner.forward(arc_scores, temp, lengths)


def eisner_full_outputs(arc_scores, temp, lengths):
    value, marginals = eisner.forward(arc_scores, temp, lengths)
    grad_temp = eisner.value_grad_params(arc_scores, temp, lengths)
    return value, marginals, grad_temp


def lev_forward_with_grads(scores, ins, delete, temp, lengths):
    value, marginals = lev.forward(
        scores, ins, delete, temp, lengths
    )
    grad_ins, grad_del, grad_temp = lev.value_grad_params(
        scores, ins, delete, temp, lengths
    )
    return value, marginals, grad_ins, grad_del, grad_temp


def lev_param_field(scores, index, ins, delete, temp, lengths):
    return _field(
        index,
        (
            lev.marginals_grad_ins_cost,
            lev.marginals_grad_del_cost,
            lev.marginals_grad_temp,
        ),
        "param_type must be 0, 1, or 2",
        scores,
        ins,
        delete,
        temp,
        lengths,
    )


def lcs_forward_with_grads(scores, temp, lengths):
    value, marginals = lcs.forward(scores, temp, lengths)
    grad_temp = lcs.value_grad_params(scores, temp, lengths)
    return value, marginals, grad_temp


def osa_forward_with_grads(
    sub_costs,
    trans_mask,
    ins,
    delete,
    trans,
    temp,
    lengths,
):
    value, marginals = osa.forward(
        sub_costs,
        trans_mask,
        ins,
        delete,
        trans,
        temp,
        lengths,
    )
    grad_ins, grad_del, grad_trans, grad_temp = osa.value_grad_params(
        sub_costs,
        trans_mask,
        ins,
        delete,
        trans,
        temp,
        lengths,
    )
    return (
        value,
        marginals,
        grad_temp,
        grad_ins,
        grad_del,
        grad_trans,
    )


def osa_param_field(
    sub_costs,
    trans_mask,
    index,
    ins,
    delete,
    trans,
    temp,
    lengths,
):
    return _field(
        index,
        (
            osa.marginals_grad_ins_cost,
            osa.marginals_grad_del_cost,
            osa.marginals_grad_trans_cost,
            osa.marginals_grad_temp,
        ),
        "param_type must be 0, 1, 2, or 3",
        sub_costs,
        trans_mask,
        ins,
        delete,
        trans,
        temp,
        lengths,
    )


def damerau_forward_with_grads(
    sub_costs,
    trans_src,
    ins,
    delete,
    trans,
    temp,
    lengths,
):
    value, marginals = damerau.forward(
        sub_costs,
        trans_src,
        ins,
        delete,
        trans,
        temp,
        lengths,
    )
    grad_ins, grad_del, grad_trans, grad_temp = (
        damerau.value_grad_params(
            sub_costs,
            trans_src,
            ins,
            delete,
            trans,
            temp,
            lengths,
        )
    )
    return (
        value,
        marginals,
        grad_temp,
        grad_ins,
        grad_del,
        grad_trans,
    )


def damerau_param_field(
    sub_costs,
    trans_src,
    index,
    ins,
    delete,
    trans,
    temp,
    lengths,
):
    return _field(
        index,
        (
            damerau.marginals_grad_ins_cost,
            damerau.marginals_grad_del_cost,
            damerau.marginals_grad_trans_cost,
            damerau.marginals_grad_temp,
        ),
        "param_type must be 0, 1, 2, or 3",
        sub_costs,
        trans_src,
        ins,
        delete,
        trans,
        temp,
        lengths,
    )
