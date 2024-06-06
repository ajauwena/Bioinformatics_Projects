#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./read_group_reindexer.sh /scratch/ajauwena/binf_6110/p3/sorted_bam_files_read_groups

# Command line arguments:
    # $0: ./read_group_reindexer.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/sorted_bam_files_read_groups (a directory containing the sorted .bam files for the 10 individual fish with read groups).

# Loop through each sorted .bam file with a read group in the appropriate directory.
for file in $1/*
do

    # Extract the file's base name.
    base_name=$(echo ${file} | rev | cut -d '.' -f2 | cut -d '/' -f1 | rev)

    # Inform the user of the file that is being processed.
    echo "Reindexing" ${base_name} "..."

    # Reindex the sorted .bam file with a read group.
    samtools index ${file}

done