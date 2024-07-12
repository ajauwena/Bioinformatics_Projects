#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=02:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Generating_A_List_of_BAM_Files_May_2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Set a variable containing the absolute path to the directory for the sorted BAM files for the May 2022 barley samples.
dir_bam_may_2022="/home/ajauwena/scratch/binf_6999/barley_may_2022/bam"

# Loop through each sorted BAM file. 
for file in ${dir_bam_may_2022}/*
do

    # Obtain the file's extension name.
    extension_name=$(echo ${file} | rev | cut -d "_" -f 1 | rev)

    # If the file's extension name is "sort.bam"...
    if [ ${extension_name} == "sort.bam" ]
    then
    
        # Append the absolute path to the file to a TXT file.
        echo ${file} >> ${dir_bam_may_2022}/bam_list_may_2022.txt
    
    fi

done