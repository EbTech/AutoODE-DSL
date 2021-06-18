"""
Tests of the ``SmartState``.
"""

from unittest import TestCase

from ode_nn.state import SmartState


class SmartStateTest(TestCase):
    def test_smoke(self):
        ss = SmartState()
        assert ss is not None

    def test_attributes(self):
        ss = SmartState("SIR", n_days=10, n_regions=42)
        assert ss.tensor.shape == (10, 3, 42)
        for k, pop in enumerate("SIR"):
            assert ss.mapping[pop] == k
            assert ss.mapping[k] == pop

    def test___getitem__(self):
        ss = SmartState("SEITURD", n_days=10, n_regions=42)
        assert ss.tensor.shape == (10, 7, 42)
        # Check population indexing
        for pop in list("SEITURD"):
            x = ss[pop]
            assert x.shape == (10, 42)
        # Check numeric indexing
        s = ss[:5]
        assert s.shape == (5, 7, 42)
        s = ss[:5, :3, 10:20]
        assert s.shape == (5, 3, 10)

    def test_N(self):
        ss = SmartState("SEITURD", n_days=10, n_regions=42)
        assert ss.N.shape == (10, 42)

    def test_requires_grad(self):
        ss = SmartState("SEITURD", n_days=10, n_regions=42, requires_grad=True)
        assert ss.tensor.requires_grad

    def test_device(self):
        ss = SmartState("SEITURD", n_days=10, n_regions=42, requires_grad=True)
        assert not ss.tensor.is_cuda
