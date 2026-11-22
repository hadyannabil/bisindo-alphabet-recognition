from bisindo_app.core import AutoConfirmGate


def test_gate_requires_continuous_stability():
    gate = AutoConfirmGate(hold_seconds=1.8)
    assert not gate.update('A', now=10.0)
    assert not gate.update('A', now=11.7)
    assert gate.update('A', now=11.8)
    assert not gate.update('A', now=15.0)


def test_gate_rearms_after_no_hand_or_new_letter():
    gate = AutoConfirmGate(hold_seconds=1)
    gate.update('A', now=1)
    assert gate.update('A', now=2)
    assert not gate.update(None, now=3)
    assert not gate.update('A', now=4)
    assert gate.update('A', now=5)
    assert not gate.update('B', now=6)
    assert gate.update('B', now=7)


def test_gate_manual_confirmation_suppresses_auto():
    gate=AutoConfirmGate(hold_seconds=1)
    gate.update('C',now=1)
    gate.mark_manual_confirm('C')
    assert not gate.update('C',now=50)
    assert not gate.update('D',now=51)
    assert gate.update('D',now=52)


def test_gate_rejects_invalid_prediction():
    gate=AutoConfirmGate(hold_seconds=1)
    assert not gate.update('aa',now=2)
    assert gate.candidate is None
