#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./variant_caller_and_vcf_file_combiner_freebayes.sh /scratch/ajauwena/binf_6110/p3/burbot_genome /scratch/ajauwena/binf_6110/p3/sorted_bam_files_read_groups /scratch/ajauwena/binf_6110/p3/results_freebayes

# Command line arguments:
    # $0: ./variant_caller_and_vcf_file_combiner_freebayes.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/burbot_genome (a directory containing the burbot reference genome).
    # $2: /scratch/ajauwena/binf_6110/p3/sorted_bam_files_read_groups (a directory containing the sorted and reindexed .bam files for the 10 individual fish with read groups).
    # $3: /scratch/ajauwena/binf_6110/p3/results_freebayes (a directory containing the results obtained from calling variants using freebayes).

# Loop through each sorted .bam file in the appropriate directory.
for file in $2/*
do

    # Extract the file's extension name.
    extension_name=$(echo ${file} | rev | cut -d '.' -f1 | rev)

    # If the file's extension name is "bam"...
    if [ ${extension_name} == "bam" ]
    then

        # Append the file to a list.
        echo ${file} >> $3/bam_file_list_freebayes.txt
    
    fi

done

# Call variants in the files in the list using freebayes.
freebayes -f $1/burbot_2021.fasta -L $3/bam_file_list_freebayes.txt > $3/results_freebayes.vcf