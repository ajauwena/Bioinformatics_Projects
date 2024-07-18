#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./sam_to_sorted_bam_converter.sh /scratch/ajauwena/binf_6110/p3/bwa_assem /scratch/ajauwena/binf_6110/p3/sorted_bam_files

# Command line arguments:
    # $0: ./sam_to_sorted_bam_converter.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/bwa_assem (a directory containing the .sam files for the 10 individual fish).
    # $2: /scratch/ajauwena/binf_6110/p3/sorted_bam_files (a directory that will contain the sorted .bam files for the 10 individual fish).

# Loop through each .sam file in the appropriate directory.
for file in $1/*
do

    # Extract the file's base name.
    base_name=$(echo ${file} | rev | cut -d '.' -f3 | cut -d '/' -f1 | rev)

    # Convert the .sam file to a sorted .bam file.
    samtools sort -O bam -o $2/${base_name}_sorted.bam ${file}

done