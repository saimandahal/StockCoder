import os
import pandas as pd
import shutil



def filter_company_files(root_folder, target_folder):
    filtered_files = []
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            #print(filename)
            if filename.endswith('.csv'):
                filepath = os.path.join(foldername, filename)
                try:
                    # Read CSV skipping first row as header
                    df = pd.read_csv(filepath)#, skiprows=1)
                    if not df.empty:
                        start_date = pd.to_datetime(df['date'].iloc[-1])
                        if start_date <= pd.Timestamp('2010-01-01'):
                            filtered_files.append(filename)
                            shutil.copy(filepath, os.path.join(target_folder, filename))

                    else:
                        print(f"Skipping file {filename} because it contains no data.")
                except KeyError:
                    print(f"Skipping file {filename} because 'date' column not found.")
    return filtered_files


root_folder = r'D:\USER FILES\DESKTOP\WSU\Spring 2024\Neural Network\Project\Dataset\Index_dataset\full_history'
target_folder = r'D:\USER FILES\DESKTOP\WSU\Spring 2024\Neural Network\Project\Dataset\Index_dataset\filtered_dataset'
filtered_files = filter_company_files(root_folder, target_folder)


print(filtered_files)