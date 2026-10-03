# SPDX-License-Identifier: Apache-2.0
"""Native inside charts agree with the existing tree operators and enumeration."""
from functools import lru_cache

import pytest
import torch
import orihime as ohm

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("algorithm",("cky","eisner"))
@pytest.mark.parametrize("n",(1,2,4))
def test_tree_value_layout_and_snapshots(device,algorithm,n):
    torch.manual_seed(634)
    shape=(2,n,n,n) if algorithm == "cky" else (2,n,n)
    x=(torch.randn(shape,device=device)*.2).requires_grad_()
    t=torch.tensor(.8,device=device,requires_grad=True)
    leaves=torch.randn(2,n,device=device,requires_grad=True)*.1 if algorithm == "cky" else None
    state=ohm.forward_messages(x,algorithm=algorithm,leaf_scores=leaves,temperature=t)
    expected=ohm.cky_value(x,leaves,temperature=t) if algorithm == "cky" else ohm.eisner_value(x,temperature=t)
    torch.testing.assert_close(state.value,expected,atol=2e-6,rtol=2e-6)
    assert state.alpha.shape == (2,1 if algorithm == "cky" else 4,n,n)
    assert state.states == (("I",) if algorithm == "cky" else ("C_R","C_L","I_R","I_L"))
    assert state.is_tree and not state.padded
    assert not state.alpha.requires_grad and not state.value.requires_grad
    assert state.inputs.requires_grad and state.temperature.requires_grad
    before=state.inputs.clone()
    with torch.no_grad():x.add_(1)
    assert torch.equal(state.inputs,before)


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("algorithm",("cky","eisner"))
def test_ragged_tree_roots(device,algorithm):
    n=5
    shape=(3,n,n,n) if algorithm == "cky" else (3,n,n)
    x=torch.randn(shape,device=device)*.1
    leaves=torch.randn(3,n,device=device)*.1 if algorithm == "cky" else None
    lengths=torch.tensor([1,3,5],device=device,dtype=torch.int32)
    state=ohm.forward_messages(x,algorithm=algorithm,leaf_scores=leaves,lengths=lengths)
    for b,l in enumerate(lengths.tolist()):
        if algorithm == "cky":expected=ohm.cky_value(x[b:b+1,:l,:l,:l].contiguous(),leaves[b:b+1,:l].contiguous())
        else:expected=ohm.eisner_value(x[b:b+1,:l,:l].contiguous())
        torch.testing.assert_close(state.value[b],expected[0],atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize("device",DEVICES)
def test_cky_nonuniform_temperatures_enumerate_local_splits(device):
    n=4
    x=torch.randn(1,n,n,n,device=device)*.1
    leaves=torch.randn(1,n,device=device)*.1
    t=torch.rand(1,n,n,device=device)+.4
    state=ohm.forward_messages(x,algorithm="cky",leaf_scores=leaves,temperature=t)
    @lru_cache(None)
    def inside(i,j):
        if i == j:return leaves[0,i]
        terms=torch.stack([inside(i,k)+inside(k+1,j)+x[0,i,k,j] for k in range(i,j)])
        return t[0,i,j]*torch.logsumexp(terms/t[0,i,j],0)
    torch.testing.assert_close(state.value[0],inside(0,n-1),atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("algorithm",("cky","eisner"))
def test_zero_batch_and_mask(device,algorithm):
    shape=(0,3,3,3) if algorithm == "cky" else (0,3,3)
    x=torch.empty(shape,device=device)
    leaves=torch.empty(0,3,device=device) if algorithm == "cky" else None
    state=ohm.forward_messages(x,algorithm=algorithm,leaf_scores=leaves)
    assert state.value.shape == (0,)
    x=torch.randn((1,*shape[1:]),device=device)*.1
    leaves=torch.zeros(1,3,device=device) if algorithm == "cky" else None
    mask=torch.zeros_like(x,dtype=torch.bool)
    if algorithm == "cky":mask[0,0,0,2]=True
    else:mask[0,0,2]=True
    state=ohm.forward_messages(x,algorithm=algorithm,leaf_scores=leaves,mask=mask)
    expected=ohm.cky_value(x,leaves,mask=mask) if algorithm == "cky" else ohm.eisner_value(x,mask=mask)
    torch.testing.assert_close(state.value,expected,atol=2e-6,rtol=2e-6)
    assert torch.isneginf(state.inputs[mask]).all()


@pytest.mark.parametrize("algorithm",("cky","eisner"))
@pytest.mark.parametrize("case",("length_dtype","length_zero","shape","temperature","leaf"))
def test_invalid_tree_contract(algorithm,case):
    x=torch.zeros((1,3,3,3) if algorithm == "cky" else (1,3,3))
    kwargs=dict(algorithm=algorithm,leaf_scores=torch.zeros(1,3) if algorithm == "cky" else None)
    if case == "length_dtype":kwargs["lengths"]=torch.tensor([3],dtype=torch.int64)
    elif case == "length_zero":kwargs["lengths"]=torch.tensor([0],dtype=torch.int32)
    elif case == "shape":x=x[..., :2].contiguous()
    elif case == "temperature":kwargs["temperature"]=0.
    else:kwargs["leaf_scores"]=torch.zeros(2,3)
    with pytest.raises((ValueError,RuntimeError)):ohm.forward_messages(x,**kwargs)
