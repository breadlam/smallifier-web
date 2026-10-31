#!/usr/bin/env python3
"""Writes fitted model parameters (fit_model.py --save output) into
crates/core/src/model.rs, at full precision (rounding can flip a large
histogram bin across a quantiser threshold and change predictions by
several percent).

  port_model.py CONST fit.json      e.g. port_model.py MEDIUM target/fit/medium.json
"""
import json
import re
import sys

name, path = sys.argv[1], sys.argv[2]
fit = json.load(open(path))
r = dict(zip(fit["rate_names"], fit["rate"]))
d = dict(zip(fit["dist_names"], fit["dist"]))
rate = (f"    rate: RateParams {{\n"
        f"        a_i: {r['aI']!r}, b_i: {r['bI']!r}, c_i: {r['cI']!r},\n"
        f"        a_p: {r['aP']!r}, b_p: {r['bP']!r}, c_p: {r['cP']!r},\n"
        f"        d_i: {r['dI']!r}, d_p: {r['dP']!r}, tau: {r['tau']!r}, tau_hf: {r['tau_hf']!r}, mv_p: {r['mvP']!r},\n"
        f"    }},\n"
        f"    dist: DistParams {{ ln_a: {d['lnA']!r}, b: {d['b']!r}, g: {d['g']!r} }},\n"
        f"    q_double: {fit.get('qd', 6.0)!r},\n")
src = open("crates/core/src/model.rs").read()
pat = re.compile(r"(pub const " + name + r": Model = Model \{\n)    rate: RateParams \{.*?\n    q_double: [^\n]*\n", re.S)
new, n = pat.subn(lambda m: m.group(1) + rate, src)
assert n == 1, f"const {name} not found"
open("crates/core/src/model.rs", "w").write(new)
print(f"{name} <- {path}")
