import pandas as pd
import itertools
from collections import Counter
from scipy.stats import pearsonr

def get_kmer_counts(sequences, k=3):
    """Counts frequency of all k-mers in a list of sequences."""
    counts = Counter()
    total_kmers = 0
    
    for seq in sequences:
        # sliding window
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            counts[kmer] += 1
            total_kmers += 1
            
    # Normalize to frequencies
    freqs = {kmer: count / total_kmers for kmer, count in counts.items()}
    return freqs

# 1. Load Data (assuming df1 and df2 from your previous code)
# df1 = Generated, df2 = Natural

path_to_file1 = "/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/train_dataset_top_5%.csv"
path_to_file2 = "/home/xli263/xli/utr_design/DRAKES/Ablation_study_mrl/generated_optimized_sequences_reverse_kl.csv"

df1 = pd.read_csv(path_to_file1)
df2 = pd.read_csv(path_to_file2)



real_seqs = df1['utr'].dropna().astype(str).tolist()  # Replace 'seq' with actual col name if 'utr'
gen_seqs = df2['seq'].dropna().astype(str).tolist()

# 2. Calculate 3-mer frequencies
k = 5
real_counts = get_kmer_counts(real_seqs, k)
gen_counts = get_kmer_counts(gen_seqs, k)

# 3. Align the data for plotting
all_kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
x_vals = [real_counts.get(kmer, 0) for kmer in all_kmers]
y_vals = [gen_counts.get(kmer, 0) for kmer in all_kmers]

# 4. Calculate Correlation
r, p_value = pearsonr(x_vals, y_vals)

print(f"Pearson r: {r:.6f}")
