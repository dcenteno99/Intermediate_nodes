import numpy as np

from inflation.inflation import InflationProblem, InflationSDP

# Although C has 4 settings, we only refer to one in the objective, so we'll lie WLOG in the input.
tripartite_Bell = InflationProblem(
    dag={"rho_ABC": ["A", "B", "C"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 2, 1],
    inflation_level_per_source=[1],
    order=['A', 'B', 'C'],
    verbose=0)

print("Internal operator layout:")
print(list(map(str, tripartite_Bell._lexrepr_to_names[::2])))

print("Before manually adjusting noncommutation relations:")
print(tripartite_Bell._default_notcomm.astype(int)[::2,::2])
n_ops = tripartite_Bell._nr_operators
# We want to impose that when A and C have different settings they do not commute.
# Our SDP only considers C to use setting 0. So A_1 and C_0 do not commute.
AC_intermediate_latent_noncomm = tripartite_Bell._default_notcomm.copy()
for i, j in np.ndindex((n_ops,n_ops)):
    op_i = tripartite_Bell._lexorder[i]
    op_j = tripartite_Bell._lexorder[j]
    if (op_i[0] == 1) and (op_j[0] == 3):
        x_A = op_i[-2]
        x_C, y_C = np.divmod(op_j[-2], 2) # Both 0, by construction
        if x_A != y_C:
            AC_intermediate_latent_noncomm[i, j] = True
            AC_intermediate_latent_noncomm[j, i] = True

print("After manually adjusting noncommutation relations:")
print(tripartite_Bell._default_notcomm.astype(int)[::2,::2])


CHSH_objective = {
    "P[A_0=0 B_0=0]": 4,
    "P[A_0=0 B_1=0]": 4,
    "P[A_1=0 B_0=0]": 4,
    "P[A_1=0 B_1=0]": -4,
    "P[A_0=0]": -4,
    "P[B_0=0]": -4,
    "1": 2}

manuscript_objective_AC= {
    "P[A_0=0 B_0=0]": 4,
    "P[A_0=0 B_1=0]": 4,
    "P[A_1=0 B_0=0]": 4,
    "P[A_1=0 B_1=0]": -4,
    "P[A_0=0]": -8,
    "P[B_0=0]": -4,
    "P[C=0]": -4,
    "P[A_0=0 C=0]": 8,
    "1": 4}

manuscript_objective_BC= {
    "P[A_0=0 B_0=0]": 4,
    "P[A_0=0 B_1=0]": 4,
    "P[A_1=0 B_0=0]": 4,
    "P[A_1=0 B_1=0]": -4,
    "P[A_0=0]": -4,
    "P[B_0=0]": -8,
    "P[C=0]": -4,
    "P[B_0=0 C=0]": 8,
    "1": 4}

manuscript_objective_AB= {
    "P[A_0=0 B_0=0]": 12,
    "P[A_0=0 B_1=0]": 4,
    "P[A_1=0 B_0=0]": 4,
    "P[A_1=0 B_1=0]": -4,
    "P[A_0=0]": -8,
    "P[B_0=0]": -8,
    "1": 4}


print("***Solving SDP with AC intermediate latent***")
tripartite_Bell._default_notcomm = AC_intermediate_latent_noncomm
one_intermediate_latent_SDP = InflationSDP(tripartite_Bell, verbose=0)
one_intermediate_latent_SDP.generate_relaxation("npa2")
print("[Polygamy objective is Charlie guessing Alice]")
one_intermediate_latent_SDP.set_objective(manuscript_objective_AC, direction="max")
one_intermediate_latent_SDP.solve(solve_dual=False)
print("Max objective:", one_intermediate_latent_SDP.primal_objective)
print("  2+2*sqrt(2):", 2+2*np.sqrt(2))
print("[Polygamy objective is Charlie guessing Bob]")
one_intermediate_latent_SDP.set_objective(manuscript_objective_BC, direction="max")
one_intermediate_latent_SDP.solve(solve_dual=False)
print("Max objective:", one_intermediate_latent_SDP.primal_objective)
print("    8/sqrt(3):", 8/np.sqrt(3))

print("[Variant CHSH where we add Alice guessing Bob to the payoff]")
one_intermediate_latent_SDP.set_objective(manuscript_objective_AB, direction="max")
one_intermediate_latent_SDP.solve(solve_dual=False)
print("Max objective:", one_intermediate_latent_SDP.primal_objective)
print("    8/sqrt(3):", 8/np.sqrt(3))