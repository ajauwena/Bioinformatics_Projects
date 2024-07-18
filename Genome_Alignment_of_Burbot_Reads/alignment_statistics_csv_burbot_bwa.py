# Before running this Python script, scp all the .txt files from your Unix directory into a directory in your local computer.

import os
import pandas as pd

# Create an empty list "list_burbot_statistics_bwa".
list_burbot_statistics_bwa = []

# Set a variable storing the absolute path to a directory in my local computer.
path_burbot_statistics_bwa = r"C:\Users\ajauw\OneDrive\Documents\m_a\w\education\master's\mbinf\w23\binf_6110\projects\p2\alignment_statistics_txt\burbot_statistics_bwa"
# Feel free to change the path above to a path to your specific directory.

# Loop through all the .txt files in the directory "burbot_statistics_bwa".
for txt_file in os.listdir(path_burbot_statistics_bwa):

    # Open the file.
    with open(os.path.join(path_burbot_statistics_bwa, txt_file)) as file:

        # Obtain the name of the file.
        name = txt_file

        # Read the line in the file (each file only contains one line).
        line = file.readlines()

        # Store the elements in each line as a list, split by the whitespace character.
        element = line[0].split(' ')

        # Obtain the number of reads mapped.
        reads_mapped = float(element[1])

        # Obtain the raw total sequences.
        raw_total_sequences = float(element[3])

        # Obtain the error rate.
        error_rate = float(element[5])

        # Obtain the average length.
        average_length = float(element[7])

        # Obtain the average quality.
        average_quality = float(element[9])

        # Obtain the alignment completeness.
        alignment_completeness = float(element[11])

        # Append the name and all the obtained statistics to the list "list_burbot_statistics_bwa".
        list_burbot_statistics_bwa.append([name, reads_mapped, raw_total_sequences, error_rate, average_length, average_quality, alignment_completeness])
        # "list_burbot_statistics_bwa" is a list of lists.

# Create a data frame "df_burbot_statistics_bwa" from the list "list_burbot_statistics_bwa".
df_burbot_statistics_bwa = pd.DataFrame(list_burbot_statistics_bwa, columns=['File_Name', 'Reads_Mapped', 'Raw_Total_Sequences', 'Error_Rate', 'Average_Length', 'Average_Quality', 'Alignment_Completeness'])

# Output the data frame "df_burbot_statistics_bwa" as a .csv file.
df_burbot_statistics_bwa.to_csv('df_burbot_statistics_bwa.csv')