import numpy as np
from itertools import product

BLOCKSIZE = 128
M = 1
R0 = 42
R0_STAR = R0 + 2


ACTIVE_ROUND = R0_STAR - 2
ACTIVE_SBOX = 0

NUM_TRIALS = 10000
VERBOSE_EVERY = 100
MAX_EXACT_NULLITY = 22
MATRIX_BACKEND = "grain"

blocksize = BLOCKSIZE
m = M
r0 = R0

# DDT allowed transitions for one active 3-bit S-box.
ACTIVE_TABLE = {
    (0, 0, 1): [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
    (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)],
    (0, 1, 1): [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0)],
    (1, 0, 0): [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
    (1, 0, 1): [(0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0)],
    (1, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)],
    (1, 1, 1): [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)],
}



def gf2_mul(A, B):
    return (A @ B) % 2


def row_bits_to_int(row):
    value = 0
    for i, bit in enumerate(row):
        value |= (int(bit) & 1) << i
    return value


def gf2_rank(A):
    basis = {}
    rank = 0

    for row in np.asarray(A):
        x = row_bits_to_int(row)
        while x:
            pivot = x.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = x
                rank += 1
                break
            x ^= existing

    return rank


def gf2_nullspace(A):
    A = A.copy() % 2
    n_rows, n_cols = A.shape
    row = 0
    pivots = []

    for col in range(n_cols):
        pivot = None
        for r in range(row, n_rows):
            if A[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]

        for r in range(n_rows):
            if r != row and A[r, col] == 1:
                A[r] ^= A[row]

        pivots.append(col)
        row += 1
        if row == n_rows:
            break

    free_cols = [c for c in range(n_cols) if c not in pivots]
    basis = []

    for free in free_cols:
        v = np.zeros(n_cols, dtype=int)
        v[free] = 1
        for i, pcol in enumerate(pivots):
            if A[i, free] == 1:
                v[pcol] = 1
        basis.append(v)

    return basis


# =========================
# Grain self-shrinking generator
# =========================
def grain_ssg(seed=None):
    if seed is None:
        state = [1 for _ in range(80)]
    else:
        if len(seed) != 80:
            raise ValueError("seed length must be 80")
        state = [int(x) & 1 for x in seed]

    index = 0

    for _ in range(160):
        state[index] ^= (
            state[(index + 13) % 80]
            ^ state[(index + 23) % 80]
            ^ state[(index + 38) % 80]
            ^ state[(index + 51) % 80]
            ^ state[(index + 62) % 80]
        )
        index = (index + 1) % 80

    while True:
        state[index] ^= (
            state[(index + 13) % 80]
            ^ state[(index + 23) % 80]
            ^ state[(index + 38) % 80]
            ^ state[(index + 51) % 80]
            ^ state[(index + 62) % 80]
        )
        choice = state[index]
        index = (index + 1) % 80

        state[index] ^= (
            state[(index + 13) % 80]
            ^ state[(index + 23) % 80]
            ^ state[(index + 38) % 80]
            ^ state[(index + 51) % 80]
            ^ state[(index + 62) % 80]
        )

        if choice == 1:
            yield state[index]

        index = (index + 1) % 80


def random_seed():
    return np.random.randint(0, 2, size=80).tolist()


def instantiate_matrix(n, m_, gen=None, backend="grain", rng=None):
    if backend == "grain":
        if gen is None:
            raise ValueError("backend='grain'  gen")

        while True:
            mat = np.fromiter(gen, dtype=np.uint8, count=n * m_).reshape(n, m_)
            if gf2_rank(mat) >= min(n, m_):
                return mat

    if backend == "random_full_rank":
        if rng is None:
            rng = np.random.default_rng()

        while True:
            mat = rng.integers(0, 2, size=(n, m_), dtype=np.uint8)
            if gf2_rank(mat) >= min(n, m_):
                return mat

    if backend == "fast_invertible":
        if n != m_:
            raise ValueError("backend='fast_invertible'")

        if rng is None:
            rng = np.random.default_rng()

        lower = np.tril(rng.integers(0, 2, size=(n, n), dtype=np.uint8), k=-1)
        upper = np.triu(rng.integers(0, 2, size=(n, n), dtype=np.uint8), k=1)
        np.fill_diagonal(lower, 1)
        np.fill_diagonal(upper, 1)
        mat = ((lower.astype(np.uint16) @ upper.astype(np.uint16)) & 1).astype(
            np.uint8, copy=False
        )
        row_perm = rng.permutation(n)
        col_perm = rng.permutation(n)
        return mat[row_perm][:, col_perm]

    raise ValueError(f"backend: {backend}")


def generate_linlayers(blocksize, rounds, backend=MATRIX_BACKEND):
    if backend == "grain":
        seed = random_seed()
        gen = grain_ssg(seed)
        return [
            instantiate_matrix(blocksize, blocksize, gen=gen, backend=backend)
            for _ in range(rounds)
        ]

    rng = np.random.default_rng()
    return [
        instantiate_matrix(blocksize, blocksize, backend=backend, rng=rng)
        for _ in range(rounds)
    ]


def sbox_slice(sbox_idx):
    start = 3 * sbox_idx
    return slice(start, start + 3)


def zero_except_sbox_rows(m, sbox_idx):
    cols = [
        j
        for j in range(3 * m)
        if j not in range(3 * sbox_idx, 3 * sbox_idx + 3)
    ]
    S = np.zeros((len(cols), 3 * m), dtype=int)
    for r, c in enumerate(cols):
        S[r, c] = 1
    return S


def validate_type_iii_parameters(active_round, active_sbox, r0_star):
    if not (0 <= active_sbox < m):
        raise ValueError(f"0 <= active_sbox < {m}")
    if not (0 <= active_round < r0_star - 1):
        raise ValueError(
            f"0 <= active_round < r0_star - 1 = {r0_star - 1}"
        )


def build_prefix_constraints(L, active_round, active_sbox):
    bottom_vars = blocksize - 3 * m
    initial_dx_vars = 3 if active_round == 0 else 0
    num_vars = bottom_vars + initial_dx_vars

    E = np.zeros((blocksize, num_vars), dtype=int)
    E[3 * m:, :bottom_vars] = np.identity(bottom_vars, dtype=int)

    if active_round == 0:
        E[sbox_slice(active_sbox), bottom_vars:bottom_vars + 3] = np.identity(3, dtype=int)

    constraint_rows = []

    for rd in range(active_round + 1):
        top = E[:3 * m, :]

        if rd == active_round:
            selector = zero_except_sbox_rows(m, active_sbox)
            selected_rows = gf2_mul(selector, top)
            constraint_rows.extend(selected_rows)
            break

        constraint_rows.extend(top.copy())
        E[:3 * m, :] = 0
        E = gf2_mul(L[rd], E)

    if not constraint_rows:
        return np.zeros((0, num_vars), dtype=int), num_vars

    return np.array(constraint_rows, dtype=int), num_vars


def simulate_type_iii_candidate(L, v, active_round, active_sbox, r0_star):
    bottom_vars = blocksize - 3 * m

    delta = np.zeros(blocksize, dtype=int)
    delta[3 * m:] = v[:bottom_vars]

    if active_round == 0:
        delta[sbox_slice(active_sbox)] = v[bottom_vars:bottom_vars + 3]

    for rd in range(active_round):
        if np.any(delta[:3 * m]):
            return False
        delta[:3 * m] = 0
        delta = gf2_mul(L[rd], delta.reshape(-1, 1)).flatten()

    top = delta[:3 * m]
    selector = zero_except_sbox_rows(m, active_sbox)
    if np.any(gf2_mul(selector, top.reshape(-1, 1))):
        return False

    dx = tuple(int(x) for x in top[sbox_slice(active_sbox)])
    if dx not in ACTIVE_TABLE:
        return False

    for dy in ACTIVE_TABLE[dx]:
        test_delta = delta.copy()
        new_top = np.zeros(3 * m, dtype=int)
        new_top[sbox_slice(active_sbox)] = np.array(dy, dtype=int)
        test_delta[:3 * m] = new_top

        test_delta = gf2_mul(L[active_round], test_delta.reshape(-1, 1)).flatten()

        for rd in range(active_round + 1, r0_star):
            if np.any(test_delta[:3 * m]):
                return False
            if rd < r0_star - 1:
                test_delta[:3 * m] = 0
                test_delta = gf2_mul(L[rd], test_delta.reshape(-1, 1)).flatten()

    return True


def has_type_III_weak_for_L_list(
    L_list,
    active_round=None,
    active_sbox=None,
    r0_star=None,
    max_exact_nullity=22,
):
    if active_round is None:
        active_round = ACTIVE_ROUND
    if active_sbox is None:
        active_sbox = ACTIVE_SBOX
    if r0_star is None:
        r0_star = R0_STAR

    validate_type_iii_parameters(active_round, active_sbox, r0_star)

    required_linlayers = r0_star - 1
    if len(L_list) < required_linlayers:
        raise ValueError(
            f"at least {required_linlayers} linear layers"
        )

    L = L_list[:required_linlayers]

    A, num_vars = build_prefix_constraints(L, active_round, active_sbox)
    rank = gf2_rank(A) if A.size else 0
    nullity = num_vars - rank

    if nullity <= 0:
        return False

    if nullity > max_exact_nullity:
        return False

    basis = gf2_nullspace(A) if A.size else [
        np.eye(num_vars, dtype=int)[i] for i in range(num_vars)
    ]

    for mask in product([0, 1], repeat=len(basis)):
        v = np.zeros(num_vars, dtype=int)
        for i, bit in enumerate(mask):
            if bit:
                v ^= basis[i]

        if np.all(v == 0):
            continue

        if simulate_type_iii_candidate(L, v, active_round, active_sbox, r0_star):
            return True

    return False


def estimate_type_III_weak_probability(
    num_trials=100,
    active_round=None,
    active_sbox=None,
    r0_star=None,
    verbose_every=10,
    max_exact_nullity=22,
    matrix_backend=MATRIX_BACKEND,
):
    if active_round is None:
        active_round = ACTIVE_ROUND
    if active_sbox is None:
        active_sbox = ACTIVE_SBOX
    if r0_star is None:
        r0_star = R0_STAR

    validate_type_iii_parameters(active_round, active_sbox, r0_star)

    hit = 0
    required_linlayers = r0_star - 1

    for t in range(num_trials):
        L_list = generate_linlayers(
            blocksize,
            required_linlayers,
            backend=matrix_backend,
        )

        found = has_type_III_weak_for_L_list(
            L_list=L_list,
            active_round=active_round,
            active_sbox=active_sbox,
            r0_star=r0_star,
            max_exact_nullity=max_exact_nullity,
        )

        if found:
            hit += 1

        if (t + 1) % verbose_every == 0:
            print(
                f"trial={t+1}, hit={hit}, "
                f"estimated_prob={hit / (t+1):.6f}"
            )

    return hit / num_trials


if __name__ == "__main__":
    print(
        f"Type-III gamma=1: r0_star={R0_STAR}, "
        f"active_round={ACTIVE_ROUND}, active_sbox={ACTIVE_SBOX}"
    )

    prob = estimate_type_III_weak_probability(
        num_trials=NUM_TRIALS,
        active_round=ACTIVE_ROUND,
        active_sbox=ACTIVE_SBOX,
        r0_star=R0_STAR,
        verbose_every=VERBOSE_EVERY,
        max_exact_nullity=MAX_EXACT_NULLITY,
        matrix_backend=MATRIX_BACKEND,
    )

    print("\nEstimated Type-III probability =", prob)
