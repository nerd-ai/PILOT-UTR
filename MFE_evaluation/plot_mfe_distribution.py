# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from io import StringIO

# # 1. Load the data
# # (Replacing string variables with actual file loading in practice)

# path_to_file1 = "/home/xli263/xli/utr_design/MFE_calculation/pretrained_generated_data_with_mfe.csv"
# path_to_file2 = "/home/xli263/xli/utr_design/MFE_calculation/validation_data_with_mfe.csv"

# # Convert to DataFrames

# # 2. Preprocessing
# # Add a label column to identify the source
# df1 = pd.read_csv(path_to_file1)
# df2 = pd.read_csv(path_to_file2)


# df1['Dataset'] = 'Generated_motif'
# df2['Dataset'] = 'Natural'
# # Combine the data
# combined_df = pd.concat([df1[['mfe', 'Dataset']], df2[['mfe', 'Dataset']]])

# # 3. Plotting
# # We will create a figure with two subplots side-by-side
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Plot 1: Histogram with KDE (Kernel Density Estimate)
# # Good for seeing the shape of the distribution
# sns.histplot(data=combined_df, x='mfe', hue='Dataset', kde=True, ax=axes[0], palette="Set2")
# axes[0].set_title('MFE Distribution Histogram')
# axes[0].set_xlabel('MFE (kcal/mol)')

# # Plot 2: Boxplot with Swarmplot
# # Good for seeing median, quartiles, and individual outliers
# sns.boxplot(data=combined_df, x='Dataset', y='mfe', ax=axes[1], palette="Set2")
# sns.swarmplot(data=combined_df, x='Dataset', y='mfe', color=".2", ax=axes[1]) # Adds actual data points
# axes[1].set_title('MFE Comparison Boxplot')
# axes[1].set_ylabel('MFE (kcal/mol)')

# plt.tight_layout()
# plt.savefig('mfe_comparison_natural_motif_generated.png')
# plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
path_to_file1 = "/home/xli263/xli/utr_design/DRAKES/drakes_rna/MFE_evaluation/validation_data_with_mfe.csv"
path_to_file2 = "/home/xli263/xli/utr_design/UTRGAN/outputs/mrl_optimized_multi_runs_5000_new_with_T_with_MFE.csv"
path_to_file3 = "/home/xli263/xli/utr_design/DRAKES/drakes_rna/MFE_evaluation/generated_optimized_4_base_vanilla_ce_0.5_with_MFE.csv"

df1 = pd.read_csv(path_to_file1)
df2 = pd.read_csv(path_to_file2)
df3 = pd.read_csv(path_to_file3)

# 2. Preprocessing
df1['Dataset'] = 'Natural'
df2['Dataset'] = 'UTRGAN Optimized'
df3['Dataset'] = 'Generated_4_base (vanilla)'
combined_df = pd.concat([df1[['mfe', 'Dataset']], df2[['mfe', 'Dataset']], df3[['mfe', 'Dataset']]])

print(f"Total sequences to plot: {len(combined_df)}") # Good for debugging

# 3. Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Histogram
# common_norm=False is important so the height represents density of THAT dataset, not total
sns.histplot(data=combined_df, x='mfe', hue='Dataset', kde=True, ax=axes[0], palette="Set2", common_norm=False)
axes[0].set_title('MFE Distribution Histogram')
axes[0].set_xlabel('MFE (kcal/mol)')

# Plot 2: Boxplot + Stripplot (The Fix)
# showfliers=False hides outliers in boxplot so we don't double-plot them with stripplot
sns.boxplot(data=combined_df, x='Dataset', y='mfe', ax=axes[1], palette="Set2", showfliers=False)

# Use stripplot instead of swarmplot.
# size=2 makes dots smaller, alpha=0.1 makes them transparent to handle overlap.
# sns.stripplot(data=combined_df, x='Dataset', y='mfe', color="black", alpha=0.1, jitter=0.25, size=2, ax=axes[1])

axes[1].set_title('MFE Comparison Boxplot')
axes[1].set_ylabel('MFE (kcal/mol)')

plt.tight_layout()
plt.savefig('/home/xli263/xli/utr_design/DRAKES/drakes_rna/MFE_evaluation/mfe_comparison_UTRGAN_vanilla_3000_new.png')
print("Plot saved.")
