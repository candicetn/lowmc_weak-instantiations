import numpy as np

# =========================
# parameters
# =========================
blocksize = 128
m = 1
num_trials = 10000
verbose_every = 10


r0_list = [44]


def gf2_mul(A, B):
    return (A @ B) % 2

def gf2_rank(A):
    A = A.copy() % 2
    n_rows, n_cols = A.shape
    rank = 0

    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if A[row, col] == 1:
                pivot = row
                break

        if pivot is None:
            continue

        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]

        for row in range(n_rows):
            if row != rank and A[row, col] == 1:
                A[row] ^= A[rank]

        rank += 1
        if rank == n_rows:
            break

    return rank

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

    # Discard first 160 bits
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

# =========================
# seed
# =========================
def random_seed():
    return np.random.randint(0, 2, size=80).tolist()


def instantiate_matrix(n, m_, gen):
    while True:
        mat = []
        for _ in range(n):
            row = [next(gen) for _ in range(m_)]
            mat.append(row)

        mat = np.array(mat, dtype=np.uint8)

        if gf2_rank(mat) >= min(n, m_):
            return mat


def generate_linlayers(blocksize, rounds):
    seed = random_seed()
    gen = grain_ssg(seed)

    linlayers = []
    for _ in range(rounds):
        linlayers.append(instantiate_matrix(blocksize, blocksize, gen))

    return linlayers



def has_type_I_solution(L_list, blocksize=128, m=1, r0=42, verbose=False):
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
        if verbose:
            print("No constraints.")
        return True, 0, num_vars

    A = np.vstack(constraint_rows)
    rank = gf2_rank(A)
    nullity = num_vars - rank

    if verbose:
        print("A.shape =", A.shape)
        print("rank =", rank)
        print("nullity =", nullity)


    return nullity > 0, rank, nullity


def estimate_type_I_probability(num_trials=100, blocksize=128, m=1, r0=42, verbose_every=10):
    hit = 0
    rank_list = []
    nullity_list = []

    for t in range(num_trials):
        L_list = generate_linlayers(blocksize, r0)

        found, rank, nullity = has_type_I_solution(
            L_list=L_list,
            blocksize=blocksize,
            m=m,
            r0=r0,
            verbose=False
        )

        if found:
            hit += 1

        rank_list.append(rank)
        nullity_list.append(nullity)

        if (t + 1) % verbose_every == 0:
            print(
                f"r0={r0}, trial={t+1}, hit={hit}, "
                f"prob={hit/(t+1):.6f}, "
                f"avg_rank={np.mean(rank_list):.3f}, "
                f"avg_nullity={np.mean(nullity_list):.3f}"
            )

    return {
        "r0": r0,
        "probability": hit / num_trials,
        "avg_rank": float(np.mean(rank_list)),
        "avg_nullity": float(np.mean(nullity_list)),
        "min_nullity": int(np.min(nullity_list)),
        "max_nullity": int(np.max(nullity_list)),
    }


def main():
    print(f"blocksize = {blocksize}, m = {m}, num_trials = {num_trials}")
    print()

    for r0 in r0_list:
        result = estimate_type_I_probability(
            num_trials=num_trials,
            blocksize=blocksize,
            m=m,
            r0=r0,
            verbose_every=verbose_every
        )

        print("Result:", result)
        print("-" * 60)

if __name__ == "__main__":
    main()