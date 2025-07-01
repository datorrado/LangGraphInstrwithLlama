import os
import pandas as pd
import glob

# Folder containing your CSV files
input_folder = '3Hour/50_*/'  # Change to your folder name
output_folder = 'summaries'
user_column = 'users'  # Change this if your user column has a different name

# Gather all CSV files in the input folder
csv_files = glob.glob(os.path.join(input_folder, 'output*.csv'))

# Read and concatenate all CSV files
dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Count the number of unique users
num_users = combined_df[user_column].values[0]

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Set the output file name
output_file = os.path.join(output_folder, f'combined_stats_{num_users}.csv')

# Save the combined dataframe to the output file
combined_df.to_csv(output_file, index=False)

print(f'Combined CSV saved as: {output_file}')