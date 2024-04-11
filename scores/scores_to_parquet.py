import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import product


score_names = ["s0", "s01", "s10", "s1", "MatrixFactorization", "User2User_CF", "Item2Item_CF", "top1users", "top10users", "top25users"]
# dtypes = {"u": int, "i": int}
# for s_name in score_names:
#     dtypes[s_name] = float

s = np.load("RecSys/interpretability/scores/s0.npy")
num_users = s.shape[0]
num_items = s.shape[1]
bs = 512


def loop_fn(t):
    c, uu = t
    print(c)
    if c <= 68:
        return
    scores = {"u": [], "i": []}
    for s_name in score_names:
        scores[s_name] = []
    for u, i in product(range(uu, min(uu+bs, num_users)), range(num_items)):
        scores["u"].append(np.int32(u))
        scores["i"].append(np.int32(i))
    for s_name in score_names:
        s = np.load(f"RecSys/interpretability/scores/{s_name}.npy")
        for u, i in zip(scores["u"], scores["i"]):
            scores[s_name].append(np.float32(s[u, i]))
    # Save scores as a DataFrame
    scores = pd.DataFrame.from_dict(scores)
    print(scores.dtypes)
    scores.to_parquet("RecSys/interpretability/dask_scores/part."+str(c)+".parquet.gz", compression="gzip", index=False)

    
loop_values = list(enumerate(range(0, num_users, bs)))

p = Pool(4)
for e in p.imap_unordered(loop_fn, loop_values):
    pass
