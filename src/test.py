import pandas as pd

df = pd.read_csv("../analysis/full_dataset.csv")

# Per-requirement average of per-diagram stds
req_consistency = df.groupby('requirement_id')[
    ['completeness_std', 'correctness_std', 'hallucination_std']
].mean()

print(req_consistency)
# This should match your judge_consistency.csv