import pandas as pd
import RNA

# 1. Configuration
input_file = "/home/xli263/xli/utr_design/DRAKES/Ablation_study_mrl/generated_optimized_sequences_reverse_kl.csv"  # Replace with your actual CSV filename
output_file = "/home/xli263/xli/utr_design/DRAKES/Ablation_study_mrl/generated_optimized_sequences_reverse_kl_MFE.csvv"  # Output CSV filename

# 2. Load the data
print(f"Reading data from {input_file}...")
df = pd.read_csv(input_file)
initial_rows = len(df)

# Drop rows with missing or non-string sequences
if 'seq'not in df.columns:
    df = df[df['utr'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()
else:
    df = df[df['seq'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()


removed = initial_rows - len(df)
if removed:
    print(f"Dropped {removed} rows with missing/invalid seq values.")

# 3. Define a function to calculate MFE
def calculate_mfe(sequence):
    """
    Calculates the Minimum Free Energy (MFE) of an RNA sequence.
    Returns the MFE value (kcal/mol).
    """
    # Normalize DNA to RNA for ViennaRNA.
    sequence = sequence.replace("T", "U")
    invalid_chars = set(sequence.upper()) - set("ACGU")
    if invalid_chars:
        raise ValueError(f"Sequence contains invalid characters: {sorted(invalid_chars)}")
    # RNA.fold returns a tuple: (dot-bracket structure, mfe)
    structure, mfe = RNA.fold(sequence)
    return mfe

# Optional: If you also want the secondary structure (dot-bracket notation)
def get_structure(sequence):
    sequence = sequence.replace("T", "U")
    invalid_chars = set(sequence.upper()) - set("ACGU")
    if invalid_chars:
        raise ValueError(f"Sequence contains invalid characters: {sorted(invalid_chars)}")
    structure, mfe = RNA.fold(sequence)
    return structure

# 4. Apply the function to the 'seq' column
print("Calculating MFE for all sequences...")
if 'seq' not in df.columns:
    df['mfe'] = df['utr'].apply(calculate_mfe)
else:
    df['mfe'] = df['seq'].apply(calculate_mfe)

average_mfe = pd.to_numeric(df['mfe'], errors='coerce').mean()
print(f"Average MFE: {average_mfe:.6f}")

# Optional: Uncomment the next line if you want to save the structure too
# df['structure'] = df['seq'].apply(get_structure)

# 5. Save the results
print(f"Saving results to {output_file}...")
df.to_csv(output_file, index=False)

print("Done!")
print(df.head()) # Show the first few rows to verify
