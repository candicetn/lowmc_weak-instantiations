import numpy as np
from itertools import combinations, product


BLOCKSIZE = 128
M = 1
R0 = 42
R0_STAR = R0 + 3
GAMMA = 1
LAST_ACTIVE_ROUND = R0_STAR - 1


blocksize = BLOCKSIZE
m = M
r0 = R0


MATRIX_BACKEND = "grain"

# Running mode:
# "fixed": check only the active positions specified by ACTIVE_SBOXES;
# "enumerate": enumerate all choices of GAMMA active positions
#              within the round interval [ENUM_START_ROUND, ENUM_END_ROUND].
SEARCH_MODE = "fixed"

# Specify active round -> active S-box(es).
# Note: R0_STAR denotes the number of rounds in Part 1; active rounds use 0-based indexing.
# Therefore, the valid active rounds are 0, 1, ..., LAST_ACTIVE_ROUND.
# Examples: {6: 0}, {6: [0, 1], 30: [2]}, {LAST_ACTIVE_ROUND: 0}
ACTIVE_SBOXES = {
    6: 0
}

# The round interval for enumeration mode is inclusive.
ENUM_START_ROUND = 12
ENUM_END_ROUND = 13

NUM_TRIALS = 10000
VERBOSE_EVERY = 20
MAX_EXACT_NULLITY = 20


# DDT allowed transitions for one 3-bit S-box
ACTIVE_TABLE = {
    (0, 0, 0): [(0, 0, 0)],
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
            raise ValueError("backend='grain' needs gen")

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
            raise ValueError("backend='fast_invertible' is only supported for square matrices")

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

def build_zero_except_selected_selector(m, sbox_idx):
    """
   (3m-3) x 3m)
    """
    cols = [j for j in range(3 * m) if j not in range(3 * sbox_idx, 3 * sbox_idx + 3)]
    S = np.zeros((3 * m - 3, 3 * m), dtype=int)
    for r, c in enumerate(cols):
        S[r, c] = 1
    return S

def build_zero_except_sboxes_selector(m, sbox_indices):
    allowed_cols = set()
    for sbox_idx in sbox_indices:
        allowed_cols.update(range(3 * sbox_idx, 3 * sbox_idx + 3))

    cols = [j for j in range(3 * m) if j not in allowed_cols]
    S = np.zeros((len(cols), 3 * m), dtype=int)
    for r, c in enumerate(cols):
        S[r, c] = 1
    return S

def normalize_active_sboxes(active_sboxes, r0_star):
    if active_sboxes is None:
        active_sboxes = ACTIVE_SBOXES

    normalized = {}

    if isinstance(active_sboxes, dict):
        iterator = active_sboxes.items()
    else:
        by_round = {}
        for rd, sb in active_sboxes:
            by_round.setdefault(rd, []).append(sb)
        iterator = by_round.items()

    for rd, sboxes in iterator:
        if isinstance(sboxes, int):
            sbox_list = [sboxes]
        else:
            sbox_list = list(sboxes)

        if not (0 <= rd < r0_star):
            raise ValueError(f"need 0 <= rd < {r0_star}")

        for sb in sbox_list:
            if not (0 <= sb < m):
                raise ValueError(f"need 0 <= sb < {m}")

        if sbox_list:
            normalized[rd] = sorted(set(sbox_list))

    if not normalized:
        raise ValueError("active_sboxes cannot be empty")

    return normalized

def iter_active_sbox_configs(start_active_round, end_active_round, gamma, m_value=None):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if start_active_round > end_active_round:
        raise ValueError("start_active_round must not be larger than end_active_round")
    if m_value is None:
        m_value = m

    positions = [
        (rd, sb)
        for rd in range(start_active_round, end_active_round + 1)
        for sb in range(m_value)
    ]

    if gamma > len(positions):
        raise ValueError("gamma is too large")

    for chosen in combinations(positions, gamma):
        config = {}
        for rd, sb in chosen:
            config.setdefault(rd, []).append(sb)
        yield {rd: sorted(sboxes) for rd, sboxes in config.items()}

# =========================
# For a fixed set of matrices, determine whether there exists a weak structure.
# active_sboxes: dict[round -> sbox_index or list[sbox_index]]
# r0_star denotes the ending round of Part 1; every active round must satisfy 0 <= rd < r0_star.
# =========================
def has_type_II_weak_for_L_list(
    L_list,
    active_sboxes,
    r0_star=None,
    max_exact_nullity=22,
):
    if r0_star is None:
        r0_star = R0_STAR

    if len(L_list) < r0_star:
        raise ValueError(
            f"at least {r0_star}"
        )

    L = L_list[:r0_star]
    active_sboxes = normalize_active_sboxes(active_sboxes, r0_star)

    active_indices = sorted(active_sboxes.keys())
    active_positions = [
        (rd, sb)
        for rd in active_indices
        for sb in active_sboxes[rd]
    ]
    round0_positions = [
        (rd, sb)
        for rd, sb in active_positions
        if rd == 0
    ]
    round0_dx_offsets = {}
    active_number = len(active_positions)
    first_act = active_indices[0]


    bottom_vars = blocksize - 3 * m
    initial_dx_vars = 3 * len(round0_positions)
    base_vars = bottom_vars + initial_dx_vars
    num_vars = base_vars + active_number * 3

    for pos_idx, position in enumerate(round0_positions):
        round0_dx_offsets[position] = bottom_vars + 3 * pos_idx

    # ---------- P_global ----------
    P_global = [np.identity(blocksize, dtype=int)]
    for i in range(1, r0_star + 1):
        P_global.append(gf2_mul(L[i - 1], P_global[i - 1]))

    constraint_rows = []

    # ---------- 1. x0_part propagation ----------
    S_x_path = {}
    curr_S = np.zeros((blocksize, base_vars), dtype=int)
    curr_S[3 * m:, :bottom_vars] = np.identity(bottom_vars, dtype=int)

    for position, offset in round0_dx_offsets.items():
        _, sb = position
        curr_S[sbox_slice(sb), offset:offset + 3] = np.identity(3, dtype=int)

    if 0 in active_indices:
        curr_S[:3 * m, :] = 0

    for t in range(r0_star - 1):
        curr_S = gf2_mul(L[t], curr_S)
        S_x_path[t + 1] = curr_S.copy()

        if (t + 1) in active_indices:
            curr_S[:3 * m, :] = 0

    # ---------- 2. each y_i propagation ----------
    Y_paths = {position: {} for position in active_positions}

    for act_r, sb in active_positions:
        curr_Y = np.zeros((blocksize, 3), dtype=int)
        curr_Y[sbox_slice(sb), :] = np.identity(3, dtype=int)

        for t in range(act_r, r0_star - 1):
            curr_Y = gf2_mul(L[t], curr_Y)
            Y_paths[(act_r, sb)][t + 1] = curr_Y.copy()

            if (t + 1) in active_indices:
                curr_Y[:3 * m, :] = 0

    # ---------- 3. constrain----------
    for round_idx in range(r0_star - 1):
        curr_t = round_idx + 1

    
        if curr_t < first_act:
            M_top = P_global[curr_t][:3 * m, :]
            M_x = M_top[:, 3 * m:]

            row = np.hstack(
                [np.hstack([
                    M_x,
                    np.zeros((3 * m, initial_dx_vars), dtype=int),
                ])] +
                [np.zeros((3 * m, 3), dtype=int)] * active_number
            )

            for rr in row:
                constraint_rows.append(rr)
            continue

        if curr_t in active_indices:
            row_parts = [S_x_path[curr_t][:3 * m, :]]

            for act_r, sb in active_positions:
                if curr_t <= act_r:
                    row_parts.append(np.zeros((3 * m, 3), dtype=int))
                else:
                    row_parts.append(Y_paths[(act_r, sb)][curr_t][:3 * m, :])

            row = np.hstack(row_parts)

            selector = build_zero_except_sboxes_selector(m, active_sboxes[curr_t])
            reduced = gf2_mul(selector, row)

            for rr in reduced:
                constraint_rows.append(rr)
            continue

        row_parts = [S_x_path[curr_t][:3 * m, :]]

        for act_r, sb in active_positions:
            if curr_t <= act_r:
                row_parts.append(np.zeros((3 * m, 3), dtype=int))
            else:
                row_parts.append(Y_paths[(act_r, sb)][curr_t][:3 * m, :])

        row = np.hstack(row_parts)

        for rr in row:
            constraint_rows.append(rr)

    if len(constraint_rows) == 0:
        return False

    A = np.array(constraint_rows, dtype=int)
    rank = gf2_rank(A)
    nullity = num_vars - rank

    if nullity <= 0:
        return False

    if nullity > max_exact_nullity:
        return False

    basis = gf2_nullspace(A)

    for mask in product([0, 1], repeat=len(basis)):
        v = np.zeros(num_vars, dtype=int)

        for i, b in enumerate(mask):
            if b:
                v ^= basis[i]

        if np.all(v == 0):
            continue

        x0_bottom = v[:bottom_vars]
        x0_dx_all = v[bottom_vars:base_vars]
        y_all = v[base_vars:]

        y_list = [
            tuple(int(x) for x in y_all[i * 3:(i + 1) * 3])
            for i in range(active_number)
        ]
        y_by_position = {
            position: y_list[pos_idx]
            for pos_idx, position in enumerate(active_positions)
        }

        delta = np.zeros(blocksize, dtype=int)
        delta[3 * m:] = x0_bottom

        for pos_idx, (_, sb) in enumerate(round0_positions):
            dx0 = x0_dx_all[3 * pos_idx:3 * (pos_idx + 1)]
            delta[sbox_slice(sb)] = dx0

        transitions = []

        for i in range(r0_star):
            if i in active_indices:
                new_top = np.zeros(3 * m, dtype=int)
                for sb in active_sboxes[i]:
                    dx = tuple(int(x) for x in delta[sbox_slice(sb)])
                    dy = y_by_position[(i, sb)]
                    transitions.append((dx, dy))
                    new_top[sbox_slice(sb)] = np.array(dy, dtype=int)
                delta[:3 * m] = new_top

            delta = gf2_mul(L[i], delta.reshape(-1, 1)).flatten()

        valid = True
        for dx, dy in transitions:
            if dx not in ACTIVE_TABLE or dy not in ACTIVE_TABLE[dx]:
                valid = False
                break

        if valid:
            return True

    return False


def any_type_II_config_for_L_list(
    L_list,
    active_sboxes=None,
    start_active_round=None,
    end_active_round=None,
    gamma=None,
    r0_star=None,
    max_exact_nullity=22,
):
    """
    This function supports two modes:
      1. If active_sboxes is specified, only the given active-position configuration
         is checked.
      2. If start_active_round, end_active_round, and gamma are specified, all
         combinations of gamma active positions within the specified round interval
         are enumerated.
    It returns (found, matched_config, checked_count).
    """
    if r0_star is None:
        r0_star = R0_STAR

    if active_sboxes is not None:
        found = has_type_II_weak_for_L_list(
            L_list=L_list,
            active_sboxes=active_sboxes,
            r0_star=r0_star,
            max_exact_nullity=max_exact_nullity,
        )
        return found, normalize_active_sboxes(active_sboxes, r0_star), 1

    if start_active_round is None or end_active_round is None or gamma is None:
        raise ValueError(
            " active_sboxes"
        )

    if not (0 <= start_active_round <= end_active_round < r0_star):
        raise ValueError(
            f" 0 <= start_active_round <= end_active_round < {r0_star}"
        )

    checked_count = 0
    for config in iter_active_sbox_configs(start_active_round, end_active_round, gamma):
        checked_count += 1
        if has_type_II_weak_for_L_list(
            L_list=L_list,
            active_sboxes=config,
            r0_star=r0_star,
            max_exact_nullity=max_exact_nullity,
        ):
            return True, config, checked_count

    return False, None, checked_count

def estimate_type_II_weak_probability(
    num_trials=100,
    active_sboxes=None,
    start_active_round=None,
    end_active_round=None,
    gamma=None,
    r0_star=None,
    verbose_every=10,
    max_exact_nullity=22,
    matrix_backend=MATRIX_BACKEND
):
    if r0_star is None:
        r0_star = R0_STAR

    if (
        active_sboxes is None
        and start_active_round is None
        and end_active_round is None
        and gamma is None
    ):
        active_sboxes = ACTIVE_SBOXES

    hit = 0
    required_linlayers = r0_star

    for t in range(num_trials):
        L_list = generate_linlayers(
            blocksize,
            required_linlayers,
            backend=matrix_backend
        )

        found, _, _ = any_type_II_config_for_L_list(
            L_list=L_list,
            active_sboxes=active_sboxes,
            start_active_round=start_active_round,
            end_active_round=end_active_round,
            gamma=gamma,
            r0_star=r0_star,
            max_exact_nullity=max_exact_nullity
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
    if SEARCH_MODE == "fixed":
        prob = estimate_type_II_weak_probability(
            num_trials=NUM_TRIALS,
            active_sboxes=ACTIVE_SBOXES,
            r0_star=R0_STAR,
            verbose_every=VERBOSE_EVERY,
            max_exact_nullity=MAX_EXACT_NULLITY,
            matrix_backend=MATRIX_BACKEND
        )
        print("\nEstimated probability for fixed active set =", prob)

    elif SEARCH_MODE == "enumerate":
        config_count = sum(
            1
            for _ in iter_active_sbox_configs(
                ENUM_START_ROUND,
                ENUM_END_ROUND,
                GAMMA,
            )
        )
        print(
            f"Enumerating {config_count} active-set configs "
            f"from rounds {ENUM_START_ROUND}..{ENUM_END_ROUND} "
            f"with gamma={GAMMA}"
        )

        prob = estimate_type_II_weak_probability(
            num_trials=NUM_TRIALS,
            start_active_round=ENUM_START_ROUND,
            end_active_round=ENUM_END_ROUND,
            gamma=GAMMA,
            r0_star=R0_STAR,
            verbose_every=VERBOSE_EVERY,
            max_exact_nullity=MAX_EXACT_NULLITY,
            matrix_backend=MATRIX_BACKEND
        )
        print("\nEstimated probability for enumerated active sets =", prob)

    else:
        raise ValueError("SEARCH_MODE must be 'fixed' or 'enumerate'")
