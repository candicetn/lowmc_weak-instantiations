import math
import numpy as np
from itertools import combinations, product

# =========================
# Parameters
# =========================
BLOCKSIZE = 128
M = 1

# The number of LowMC rounds that should be generated.
TARGET_ROUNDS = 192

ENABLE_TYPE_III_DETECTION = True
TYPE_III_MIN_LOG2_P = -30
MAX_EXACT_NULLITY = 20
OUTPUT_FILENAME = "matrices_and_constants.dat"
VERBOSE = False


def determine_r0(blocksize, m):
    return blocksize // (3 * m)


R0 = determine_r0(BLOCKSIZE, M)

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


def sbox_slice(sbox_idx):
    start = 3 * sbox_idx
    return slice(start, start + 3)


def gf2_is_consistent(A, b):
    if A.size == 0:
        return not np.any(b)

    b_col = np.asarray(b, dtype=int).reshape(-1, 1) % 2
    return gf2_rank(A) == gf2_rank(np.hstack([A % 2, b_col]))


def normalize_active_sboxes(active_sboxes, blocksize, m, final_round):
    if active_sboxes is None:
        active_sboxes = {final_round - 1: list(range(m))}

    normalized = {}
    for rd, sboxes in active_sboxes.items():
        if not (0 <= rd < final_round):
            raise ValueError(
                f"active round {rd} is out of range; it must satisfy 0 <= rd < {final_round}"
            )

        sbox_list = [sboxes] if isinstance(sboxes, int) else list(sboxes)
        if not sbox_list:
            continue

        for sb in sbox_list:
            if not (0 <= sb < m):
                raise ValueError(f"S-box index {sb} is out of range; it must satisfy 0 <= sb < {m}")

        normalized[rd] = sorted(set(sbox_list))

    if not normalized:
        raise ValueError("active_sboxes must not be empty")

    if 3 * m > blocksize:
        raise ValueError("3 * m must not be larger than blocksize")

    return normalized


def ddt_transition_options():
    options = []
    for dx, dy_values in ACTIVE_TABLE.items():
        for dy in dy_values:
            options.append((dx, dy))
    return options


def ddt_direction_basis(dx):
    dy_values = ACTIVE_TABLE[dx]
    if len(dy_values) == 1:
        return []

    base = np.array(dy_values[0], dtype=int)
    diffs = [np.array(dy, dtype=int) ^ base for dy in dy_values[1:]]
    basis = gf2_nullspace(np.array(diffs, dtype=int))

    if len(basis) == 1:
        normal = basis[0]
        return [
            v for v in diffs
            if np.any(v) and int(np.dot(v, normal) % 2) == 0
        ][:2]

    return diffs[:2]


def kills_ddt_direction(coeff_block, dx):
    for direction in ddt_direction_basis(dx):
        if np.any(gf2_mul(coeff_block, direction.reshape(3, 1))):
            return False
    return True


def type_iii_log2_p3_for_config(active_sboxes, blocksize, m, r0, r0_star):
    active_sboxes = {
        rd: list(sboxes)
        for rd, sboxes in sorted(active_sboxes.items())
        if len(sboxes) > 0
    }
    active_rounds = sorted(active_sboxes)

    if not active_rounds:
        raise ValueError("active_sboxes must not be empty")

    gamma = sum(len(active_sboxes[rd]) for rd in active_rounds)
    sigma = 0
    prefix_active = 0

    for idx in range(len(active_rounds) - 1):
        rd = active_rounds[idx]
        next_rd = active_rounds[idx + 1]
        prefix_active += len(active_sboxes[rd])
        sigma += (
            (-6 * m * prefix_active - 3 * m) * (next_rd - rd)
            + 3 * len(active_sboxes[next_rd])
        )

    first_active = active_rounds[0]
    last_active = active_rounds[-1]

    return (
        sigma
        + (-6 * gamma * m - 3 * m) * (r0_star - 1 - last_active)
        - 3 * m * max(first_active - r0, 0)
    )


def type_iii_probability_from_log2(log2_p3, blocksize, m, r0, first_active):
    log2_N = blocksize - 3 * m * min(first_active, r0)

    if log2_p3 > -1e-12:
        return 1.0

    p = 2.0 ** log2_p3
    N = 2.0 ** log2_N
    return -math.expm1(N * math.log1p(-p))


def generate_type_iii_candidate_configs(
    blocksize,
    m,
    r0,
    total_rounds,
    min_log2_p=TYPE_III_MIN_LOG2_P,
):
    if min_log2_p is None:
        raise ValueError("min_log2_p must be set to terminate candidate generation")
    if total_rounds <= r0:
        raise ValueError("TARGET_ROUNDS must be larger than r0 to check Type-III candidates")

    candidates = []
    max_gamma = total_rounds - r0

    for gamma in range(1, max_gamma + 1):
        final_round = r0 + gamma
        r0_star = final_round + 1
        latest_active_round = final_round - 1
        rounds_needed = (gamma + m - 1) // m

        active_counts = []
        remaining = gamma
        for _ in range(rounds_needed):
            count = min(m, remaining)
            active_counts.append(count)
            remaining -= count
        active_counts.reverse()

        gamma_has_candidate = False

        for offset in range(final_round):
            first_round = latest_active_round - rounds_needed + 1 - offset
            if first_round < 0:
                break

            template_active_sboxes = {
                first_round + step: list(range(count))
                for step, count in enumerate(active_counts)
            }
            dominant_log2_p3 = type_iii_log2_p3_for_config(
                active_sboxes=template_active_sboxes,
                blocksize=blocksize,
                m=m,
                r0=r0,
                r0_star=r0_star,
            )

            if dominant_log2_p3 < min_log2_p:
                if offset == 0:
                    return candidates
                break

            per_round_choices = []
            for step, count in enumerate(active_counts):
                rd = first_round + step
                choices = [
                    (rd, list(sboxes))
                    for sboxes in combinations(range(m), count)
                ]
                per_round_choices.append(choices)

            for round_choice in product(*per_round_choices):
                active_sboxes = {rd: sboxes for rd, sboxes in round_choice}
                log2_p3 = type_iii_log2_p3_for_config(
                    active_sboxes=active_sboxes,
                    blocksize=blocksize,
                    m=m,
                    r0=r0,
                    r0_star=r0_star,
                )
                probability = type_iii_probability_from_log2(
                    log2_p3=log2_p3,
                    blocksize=blocksize,
                    m=m,
                    r0=r0,
                    first_active=first_round,
                )

                candidates.append({
                    "gamma": gamma,
                    "type_iii_rounds": final_round,
                    "active_sboxes": active_sboxes,
                    "log2_p3": log2_p3,
                    "estimated_probability": probability,
                })
                gamma_has_candidate = True

        if not gamma_has_candidate:
            break

    candidates.sort(key=lambda item: item["log2_p3"], reverse=True)
    return candidates


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


def instantiate_matrix(n, m_, gen):
    while True:
        mat = np.fromiter(gen, dtype=np.uint8, count=n * m_).reshape(n, m_)
        if gf2_rank(mat) >= min(n, m_):
            return mat


def generate_linlayers(blocksize, rounds):
    seed = random_seed()
    gen = grain_ssg(seed)
    return [
        instantiate_matrix(blocksize, blocksize, gen=gen)
        for _ in range(rounds)
    ]


def write_linear_layers(filename, linlayers, blocksize, rounds):
    with open(filename, "w") as matfile:
        s = (
            "LowMC matrices and constants\n"
            "============================\n"
            "Block size: " + str(blocksize) + "\n"
            "Rounds: " + str(rounds) + "\n\n"
        )
        matfile.write(s)

        s = (
            "Linear layer matrices\n"
            "---------------------"
        )
        matfile.write(s)

        for r in range(rounds):
            s = "\nLinear layer " + str(r + 1) + ":\n"
            for row in linlayers[r]:
                s += str([int(bit) for bit in row]) + "\n"
            matfile.write(s)


def has_type_I_solution(L_list, blocksize=128, m=1, r0=None):
    if r0 is None:
        r0 = determine_r0(blocksize, m)

    L = L_list[:r0]
    num_vars = blocksize - 3 * m
    constraint_rows = []

    for round_idx in range(r0 - 1):
        M = L[0].copy()
        for i in range(1, round_idx + 1):
            M = gf2_mul(L[i], M)

        M_top = M[:3 * m, :]
        M_reduced = M_top[:, 3 * m:]
        constraint_rows.append(M_reduced)

    if len(constraint_rows) == 0:
        return True, 0, num_vars

    A = np.vstack(constraint_rows)
    rank = gf2_rank(A)
    nullity = num_vars - rank

    return nullity > 0, rank, nullity


def has_type_III_weak_for_L_list(
    L_list,
    active_sboxes=None,
    blocksize=128,
    m=1,
    r0=None,
    final_round=None,
    max_exact_nullity=22,
):
    if r0 is None:
        r0 = determine_r0(blocksize, m)

    if final_round is None:
        final_round = r0 + 2
    if len(L_list) < final_round:
        raise ValueError(
            f"Type-III detection requires at least {final_round} linear layers, "
            f"but only {len(L_list)} are currently available"
        )

    L = L_list[:final_round]
    active_sboxes = normalize_active_sboxes(
        active_sboxes=active_sboxes,
        blocksize=blocksize,
        m=m,
        final_round=final_round,
    )

    active_positions = [
        (rd, sb)
        for rd in sorted(active_sboxes)
        for sb in active_sboxes[rd]
    ]

    y_offsets = {}
    bottom_vars = blocksize - 3 * m
    num_vars = bottom_vars + 3 * len(active_positions)

    for pos_idx, position in enumerate(active_positions):
        y_offsets[position] = bottom_vars + 3 * pos_idx

    E = np.zeros((blocksize, num_vars), dtype=int)
    E[3 * m:, :bottom_vars] = np.identity(bottom_vars, dtype=int)

    constraint_rows = []
    transition_checks = []
    previous_y_blocks = []
    independence_checks = []

    for round_idx in range(final_round + 1):
        allowed_sboxes = active_sboxes.get(round_idx, [])
        allowed_set = set(allowed_sboxes)

        for transition_idx, cols in previous_y_blocks:
            coeff_block = E[:3 * m, cols]
            if gf2_rank(coeff_block) > 1:
                return False
            independence_checks.append((transition_idx, coeff_block.copy()))

        if round_idx == final_round:
            constraint_rows.extend(E[:3 * m, :])
            break

        for sb in range(m):
            rows = E[sbox_slice(sb), :]
            if sb in allowed_set:
                offset = y_offsets[(round_idx, sb)]
                dy_rows = np.zeros((3, num_vars), dtype=int)
                dy_rows[:, offset:offset + 3] = np.identity(3, dtype=int)
                transition_checks.append((rows.copy(), dy_rows))
            else:
                constraint_rows.extend(rows)

        after_sbox = np.zeros((blocksize, num_vars), dtype=int)
        after_sbox[3 * m:, :] = E[3 * m:, :]

        for sb in allowed_sboxes:
            offset = y_offsets[(round_idx, sb)]
            after_sbox[sbox_slice(sb), offset:offset + 3] = np.identity(3, dtype=int)
            transition_idx = len([
                pos for pos in active_positions
                if pos[0] < round_idx or (pos[0] == round_idx and pos[1] <= sb)
            ]) - 1
            previous_y_blocks.append((transition_idx, list(range(offset, offset + 3))))

        E = gf2_mul(L[round_idx], after_sbox)

    if not constraint_rows or not transition_checks:
        return False

    A = np.array(constraint_rows, dtype=int)
    nullity = num_vars - gf2_rank(A)

    if nullity <= 0:
        return False

    options = ddt_transition_options()

    for choices in product(options, repeat=len(transition_checks)):
        if any(dx == (0, 0, 0) for dx, _ in choices):
            continue

        valid_independence = True
        for transition_idx, coeff_block in independence_checks:
            dx, _ = choices[transition_idx]
            if not kills_ddt_direction(coeff_block, dx):
                valid_independence = False
                break

        if not valid_independence:
            continue

        rows = [A]
        rhs = [np.zeros(A.shape[0], dtype=int)]

        for (dx_rows, dy_rows), (dx, dy) in zip(transition_checks, choices):
            rows.append(dx_rows)
            rhs.append(np.array(dx, dtype=int))
            rows.append(dy_rows)
            rhs.append(np.array(dy, dtype=int))

        candidate_A = np.vstack(rows)
        candidate_b = np.concatenate(rhs)

        if gf2_is_consistent(candidate_A, candidate_b):
            return True

    return False


def is_weak_matrix_set(
    L_list,
    blocksize=128,
    m=1,
    r0=None,
    active_configs=None,
    max_exact_nullity=22,
    enable_type_iii_detection=True,
):
    if r0 is None:
        r0 = determine_r0(blocksize, m)

    if enable_type_iii_detection and active_configs is None:
        active_configs = generate_type_iii_candidate_configs(
            blocksize=blocksize,
            m=m,
            r0=r0,
            total_rounds=len(L_list),
            min_log2_p=TYPE_III_MIN_LOG2_P,
        )
    elif not enable_type_iii_detection:
        active_configs = []

    type_i_rounds = r0 + 1
    type_i_found, rank, nullity = has_type_I_solution(
        L_list=L_list,
        blocksize=blocksize,
        m=m,
        r0=type_i_rounds,
    )

    if type_i_found:
        return {
            "weak": True,
            "accepted": False,
            "terminated_by": "Type-I",
            "type_I": True,
            "type_III": False,
            "type_III_config": None,
            "type_III_candidates": active_configs,
            "type_I_rounds": type_i_rounds,
            "type_I_rank": rank,
            "type_I_nullity": nullity,
        }

    if not enable_type_iii_detection:
        return {
            "weak": False,
            "accepted": True,
            "terminated_by": None,
            "type_I": False,
            "type_III": False,
            "type_III_config": None,
            "type_III_candidates": [],
            "type_III_rounds": None,
            "type_I_rounds": type_i_rounds,
            "type_I_rank": rank,
            "type_I_nullity": nullity,
        }

    type_iii_found = False
    type_iii_config = None

    for config in active_configs:
        found = has_type_III_weak_for_L_list(
            L_list=L_list,
            active_sboxes=config["active_sboxes"],
            blocksize=blocksize,
            m=m,
            r0=r0,
            final_round=config["type_iii_rounds"],
            max_exact_nullity=max_exact_nullity,
        )
        if found:
            type_iii_found = True
            type_iii_config = config
            break

    return {
        "weak": type_iii_found,
        "accepted": not type_iii_found,
        "terminated_by": "Type-III" if type_iii_found else None,
        "type_I": False,
        "type_III": type_iii_found,
        "type_III_config": type_iii_config,
        "type_III_candidates": active_configs,
        "type_III_rounds": type_iii_config["type_iii_rounds"] if type_iii_config else None,
        "type_I_rounds": type_i_rounds,
        "type_I_rank": rank,
        "type_I_nullity": nullity,
    }


def generate_and_check_matrix_set(
    blocksize=128,
    m=1,
    r0=None,
    total_rounds=None,
    active_configs=None,
    max_exact_nullity=22,
    enable_type_iii_detection=True,
):
    if r0 is None:
        r0 = determine_r0(blocksize, m)

    required_linlayers = r0 + 1
    if total_rounds is None:
        total_rounds = required_linlayers
    if total_rounds < required_linlayers:
        raise ValueError(
            f"TARGET_ROUNDS must be at least r0 + 1 for Type-I detection: "
            f"TARGET_ROUNDS={total_rounds}, r0+1={required_linlayers}"
        )

    if enable_type_iii_detection and active_configs is None:
        active_configs = generate_type_iii_candidate_configs(
            blocksize=blocksize,
            m=m,
            r0=r0,
            total_rounds=total_rounds,
            min_log2_p=TYPE_III_MIN_LOG2_P,
        )
    elif not enable_type_iii_detection:
        active_configs = []

    L_list = generate_linlayers(blocksize=blocksize, rounds=total_rounds)

    result = is_weak_matrix_set(
        L_list=L_list,
        blocksize=blocksize,
        m=m,
        r0=r0,
        active_configs=active_configs,
        max_exact_nullity=max_exact_nullity,
        enable_type_iii_detection=enable_type_iii_detection,
    )

    result["L_list"] = L_list
    result["required_linlayers"] = required_linlayers
    result["generated_linlayers"] = total_rounds
    return result


def main():
    if ENABLE_TYPE_III_DETECTION:
        active_configs = generate_type_iii_candidate_configs(
            blocksize=BLOCKSIZE,
            m=M,
            r0=R0,
            total_rounds=TARGET_ROUNDS,
            min_log2_p=TYPE_III_MIN_LOG2_P,
        )
    else:
        active_configs = []

    if VERBOSE:
        print(
            f"blocksize={BLOCKSIZE}, m={M}, r0={R0}, "
            f"TARGET_ROUNDS={TARGET_ROUNDS}, "
            f"Type-III detection={ENABLE_TYPE_III_DETECTION}"
        )

    result = generate_and_check_matrix_set(
        blocksize=BLOCKSIZE,
        m=M,
        r0=R0,
        total_rounds=TARGET_ROUNDS,
        active_configs=active_configs,
        max_exact_nullity=MAX_EXACT_NULLITY,
        enable_type_iii_detection=ENABLE_TYPE_III_DETECTION,
    )

    if result["terminated_by"] == "Type-I":
        print("weak: Type-I")
        return

    if result["terminated_by"] == "Type-III":
        print("weak: Type-III")
        if VERBOSE:
            print("matched Type-III config:", result["type_III_config"])
        return

    write_linear_layers(
        filename=OUTPUT_FILENAME,
        linlayers=result["L_list"],
        blocksize=BLOCKSIZE,
        rounds=TARGET_ROUNDS,
    )
    print("not weak")
    print(f"written: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
