# SPDX-License-Identifier: Apache-2.0
"""The forward-only table contract, across the native grid backends."""
import math

import pytest
import torch
import orihime as ohm
from orihime.messages import GRID_ALGORITHMS

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("algorithm", GRID_ALGORITHMS)
def test_value_layout_and_orientation(algorithm, device):
    x = torch.linspace(-.3,.6,24,device=device).reshape(2,4,3).requires_grad_()
    lengths = torch.tensor([[4,3],[3,2]],device=device,dtype=torch.int32)
    kwargs = dict(temperature=.7,lengths=lengths)
    if algorithm == "osa":
        allowed = torch.zeros_like(x,dtype=torch.bool)
        allowed[:,1,1] = True
        kwargs["allowed_transpositions"] = allowed
    if algorithm == "damerau":
        sources = torch.full((*x.shape,2),-1,device=device,dtype=torch.int32)
        sources[:,1,1] = 0
        kwargs["transposition_sources"] = sources
    messages = ohm.forward_messages(x,algorithm=algorithm,**kwargs)
    expected = getattr(ohm,algorithm+"_value")(x,**kwargs)
    torch.testing.assert_close(messages.value,expected,atol=2e-6,rtol=2e-6)
    planes = 3 if algorithm in ("sw_affine","sv","sv_affine","nw_affine") else 1
    assert messages.alpha.shape == ((2,1,4,3) if algorithm == "mas" else (2,planes,5,4))
    assert messages.alpha.is_contiguous()
    assert messages.layout_version == 1
    assert messages.padded == (algorithm != "mas")
    assert messages.orientation == ("cost" if algorithm in ("dtw","lev","osa","damerau") else "score")
    assert not messages.alpha.requires_grad and not messages.value.requires_grad
    # An external autograd wrapper still has the original differentiable inputs.
    assert messages.inputs.requires_grad
    messages.inputs.sum().backward()
    assert torch.equal(x.grad,torch.ones_like(x))


@pytest.mark.parametrize("device", DEVICES)
def test_sw_prefix_messages_are_soft_bellman_values(device):
    x = torch.tensor([[[.2,2.],[1.,1.5]]],device=device)
    out = ohm.forward_messages(x,algorithm="sw",gap_score=-1.,temperature=1.)
    h = out.alpha[0,0]
    assert h[0,0] == 0
    assert h[1,1].item() == pytest.approx(.2,abs=1e-6)
    assert h[1,2].item() == pytest.approx(math.log(math.exp(2.)+math.exp(-.8)),abs=1e-6)
    expected = torch.logsumexp(torch.stack((x[0,1,1],h[1,1]+x[0,1,1],h[1,2]-1,h[2,1]-1)),0)
    torch.testing.assert_close(h[2,2],expected)


@pytest.mark.parametrize("device", DEVICES)
def test_metadata_is_snapshotted(device):
    x = torch.zeros(1,2,2,device=device)
    lengths = torch.tensor([[2,2]],dtype=torch.int32,device=device)
    t = torch.tensor(1.,device=device,requires_grad=True)
    out = ohm.forward_messages(x,algorithm="nw",lengths=lengths,temperature=t)
    x.fill_(2); lengths.zero_()
    with torch.no_grad():
        t.fill_(2)
    assert torch.equal(out.inputs,torch.zeros_like(x))
    assert out.lengths.tolist() == [[2,2]]
    assert out.temperature == 1
    assert out.temperature.requires_grad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("algorithm", GRID_ALGORITHMS)
def test_masked_value_matches_public_operator(algorithm, device):
    x = torch.zeros(1,3,3,device=device)
    mask = torch.zeros_like(x,dtype=torch.bool)
    mask[:,0,2] = True
    kwargs = dict(mask=mask,temperature=.5)
    out = ohm.forward_messages(x,algorithm=algorithm,**kwargs)
    torch.testing.assert_close(out.value,getattr(ohm,algorithm+"_value")(x,**kwargs))
    assert torch.isfinite(out.value).all()
    if algorithm in ("dtw","lev","osa","damerau"):
        assert torch.isposinf(out.inputs[0,0,2])
    else:
        assert torch.isneginf(out.inputs[0,0,2])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("algorithm", GRID_ALGORITHMS)
def test_empty_batch(algorithm,device):
    out = ohm.forward_messages(torch.empty(0,3,3,device=device),algorithm=algorithm)
    assert out.value.shape == (0,)


@pytest.mark.parametrize("kwargs,match", [
    ({"algorithm":"cky"},"algorithm"),
    ({"temperature":0},"positive"),
    ({"temperature":float("nan")},"finite"),
    ({"gap_score":100.},"80"),
    ({"lengths":torch.tensor([[3,2]],dtype=torch.int32)},"lengths"),
    ({"lengths":torch.tensor([[2,2]])},"int32"),
    ({"algorithm":"mas","lengths":torch.zeros(1,2,dtype=torch.int32)},"token length"),
    ({"algorithm":"sw","bandwidth":1},"DTW"),
])
def test_invalid_contract(kwargs,match):
    with pytest.raises((ValueError,RuntimeError),match=match):
        ohm.forward_messages(torch.zeros(1,2,2),**kwargs)


def test_raw_dispatch_checks_unsafe_topology():
    x = torch.zeros(1,2,2)
    lengths = torch.tensor([[2,2]],dtype=torch.int32)
    bad = torch.full((1,2,2,2),100,dtype=torch.int32)
    with pytest.raises(RuntimeError,match="earlier"):
        torch.ops.orihime.grid_forward_messages(x,lengths,bad,10,1.,1.,1.,1.,-1)
