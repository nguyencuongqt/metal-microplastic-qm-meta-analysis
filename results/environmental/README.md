# Environmental screening output

`08_rwater_asia_summary.csv` is the current environmental-context table. The corresponding figure can be regenerated locally with `scripts/08_environmental_context.py`.

- `C_water` is the mean total waterborne metal concentration for Asian rivers and lakes reported in Table 3 of Zhou et al. (2020), *Global Ecology and Conservation* 22, e00925. DOI: 10.1016/j.gecco.2020.e00925.
- EMVP is evaluated at `C_MP = 0.01 mg/L` using the primary-model posterior at the fixed baseline (aged polystyrene, mean temperature).
- `R_water = EMVP / C_water` is a mass-balance contextual ratio. It is not a risk quotient and does not represent dissolved-phase regulatory criteria.
- The displayed EMVP and mean-ratio values follow the rounded values reported in Table S4. The 95% HDIs were propagated from the primary-model posterior.
