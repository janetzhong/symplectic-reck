"""Choose the order of adjacent-mode nulling gates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NullingStep:
    index: int
    probe: int  # input mode whose column is being cleared
    target: int  # output mode whose block (target, probe) is nulled
    anchor: int  # output mode the gate mixes with the target (anchor = target - 1)


def nulling_schedule(N):
    """Clear each input column, starting with the highest output-mode index.

    For N=4, the one-based mode pairs are (4,3), (3,2), (2,1), (4,3),
    (3,2), and (4,3)."""
    steps = []
    for probe in range(N - 1):
        for target in range(N - 1, probe, -1):
            steps.append(NullingStep(len(steps), probe, target, target - 1))
    return steps
