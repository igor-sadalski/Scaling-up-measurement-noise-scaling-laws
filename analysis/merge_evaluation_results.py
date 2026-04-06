import pandas as pd

# Load evaluation results (new)
df_new = pd.read_csv(
    "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-02_19-54_evaluate_mid_quality_models_on_full_data/evaluation_results.csv"
)

# Load original results (for lmi_original)
df_original = pd.read_csv(
    "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/collect_mi_results.csv"
)

# Function to extract signal name from filename
def extract_signal_name(filename):
    """Extract signal name from filename like 'Y_protein_counts_1.0.csv' -> 'protein_counts'
    Also handles Geneformer format: 'Y_clone_1.0_geneformer.csv' -> 'clone'
    """
    if pd.isna(filename) or filename == '':
        return None
    # Remove 'Y_' prefix
    if filename.startswith('Y_'):
        signal = filename[2:]  # Remove 'Y_' prefix
        # Remove suffix patterns like '_1.0.csv' or '_1.0_geneformer.csv'
        # Pattern: signal_name_1.0[_geneformer].csv
        if signal.endswith('.csv'):
            signal = signal[:-4]  # Remove '.csv'
            # Remove trailing patterns like '_1.0' or '_1.0_geneformer'
            if '_' in signal:
                # Split by '_' and find where the numeric part starts (like '1.0')
                parts = signal.split('_')
                # Find the index where we have a pattern like '1.0' or similar
                # Usually the signal name is everything before the last numeric part
                # For 'clone_1.0' -> 'clone', for 'clone_1.0_geneformer' -> 'clone'
                result_parts = []
                for part in parts:
                    # Check if this part looks like a version number (contains digits and dots)
                    if any(c.isdigit() for c in part) and ('.' in part or part.isdigit()):
                        break  # Stop at version number
                    result_parts.append(part)
                if result_parts:
                    return '_'.join(result_parts)
                return signal
            return signal
    return filename.replace('.csv', '')

# Standardize column names for lookup
column_map = {
    "signal": "signal_name",  # Keep original signal column as signal_name
    "mi_value": "lmi_original"
}
df_original = df_original.rename(columns=column_map)

# Restrict allowed signals (but keep all in df_new for plotting/later)
allowed_signals = [
    "Y_protein_counts_1.0.csv",
    "Y_clone_1.0.csv",
    "Y_author_day_1.0.csv",
    "Y_ng_idx_1.0.csv"
]

# Extract signal names from allowed signals for matching
allowed_signal_names = [extract_signal_name(sig) for sig in allowed_signals]

# For matching lookups
lookup_keys = ['dataset', 'size', 'quality', 'algorithm']

# Prepare the lookup table from the original (only seed=42 for correspondence)
df_original_seed42 = df_original[df_original['seed'] == 42].copy()

# Add extracted signal name to df_new for matching
df_new = df_new.copy()
df_new['signal_name'] = df_new['signal_file'].apply(extract_signal_name)

# Now merge the data properly
# First, merge df_new with df_original_seed42 to get lmi_original
merged = df_new.merge(
    df_original_seed42[lookup_keys + ['signal_name', 'lmi_original']],
    on=lookup_keys + ['signal_name'],
    how='left',
    suffixes=('', '_orig')
)

# Filter to only allowed signals
merged = merged[merged['signal_file'].isin(allowed_signals) | merged['signal_file'].isna()]

# Rename lmi column to lmi_target for clarity
merged = merged.rename(columns={'lmi': 'lmi_target'})

# Select and order columns
final_columns = ['dataset', 'size', 'quality', 'algorithm', 'model_name', 'signal_file', 
                 'signal_name', 'lmi_target', 'lmi_original', 'status', 'seed']
final_columns = [col for col in final_columns if col in merged.columns]

final = merged[final_columns].copy()

print(f"Total rows in final: {len(final)}")
print(f"Rows with lmi_original: {final['lmi_original'].notna().sum()}")
print(f"Rows with lmi_target: {final['lmi_target'].notna().sum()}")

final

