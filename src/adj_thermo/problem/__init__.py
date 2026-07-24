from __future__ import annotations

from adj_thermo.problem.base import ProblemSpec


def make_problem(name: str, **kwargs) -> ProblemSpec:
    key = str(name).lower().replace("-", "_")
    if key != "ala2" and kwargs:
        raise ValueError(f"Problem {name!r} does not accept configuration options: {sorted(kwargs)}")
    if key == "dw1d":
        from adj_thermo.problem.dw1d import make_problem as make
        return make()
    if key == "dw2d":
        from adj_thermo.problem.dw2d import make_problem as make
        return make()
    if key == "mb2d":
        from adj_thermo.problem.mb2d import make_problem as make
        return make()
    if key == "dw4":
        from adj_thermo.problem.dw4 import make_problem as make
        return make()
    if key == "lj13":
        from adj_thermo.problem.lj13 import make_problem as make
        return make()
    if key == "ala2":
        from adj_thermo.problem.ala2 import make_problem as make
        return make(**kwargs)
    raise ValueError(f"Unknown problem {name!r}. Choose one of: dw1d, dw2d, mb2d, dw4, lj13, ala2.")


__all__ = ["ProblemSpec", "make_problem"]
