# %% ===========================================================

import sympy


# %% ===========================================================
# Proste użycie równań symbolicznych

# Vin - Vs = Z1 * I
# Vs - Vout = Z2 * I
# Vout = -A * Vs

Vin, Vs, Vout, A, Z1, Z2, I = sympy.symbols('Vin Vs Vout A Z1 Z2 I')

eq1 = sympy.Eq(Vin - Vs, Z1 * I)
eq2 = sympy.Eq(Vs - Vout, Z2 * I)
eq3 = sympy.Eq(Vout, -A * Vs)

solution = sympy.solve((eq1, eq2, eq3), (Vout, Vs, I))

print(sympy.pretty(solution))


# %% ===========================================================
# Wykorzystanie modułu do rozwiązywania układów równań liniowych

# Układ równań w formie macierzowej M * z = b
# 1 * Vs + Z1 * I + 0 * Vout = Vin
# 1 * Vs - Z2 * I - 1 * Vout = 0
# A * Vs + 0 * I  + 1 * Vout = 0

Vin, Vs, Vout, A, Z1, Z2, I = sympy.symbols('Vin Vs Vout A Z1 Z2 I')

M = sympy.Matrix([
    [1,  Z1,  0],
    [1, -Z2, -1],
    [A,  0,   1]
])

b = sympy.Matrix([Vin, 0, 0])

x = M.solve(b)

solution = {Vs: x[0], I: x[1], Vout: x[2]}
print(sympy.pretty(solution))


# %% ===========================================================
# Żyrator

U, Z1, Z2, Z3, I1, I2, V1, V2, A = sympy.symbols('U Z1 Z2 Z3 I1 I2 V1 V2 A')

M = sympy.Matrix([
    [Z1, 0, 1, 0],
    [0, Z2, 0, 1],
    [0, Z3, 0, -1],
    [0, 0, 1+A, -A]
])

b = sympy.Matrix([U, U, 0, 0])

x = M.solve(b)
x = [sympy.simplify(expr) for expr in x]

solution = {I1: x[0], I2: x[1], V1: x[2], V2: x[3]}
for var, expr in solution.items():
    print(f"{var} = ")
    print(sympy.pretty(expr))
    print()

I = solution[I1] + solution[I2]
print("I = ")
print(sympy.pretty(I))

