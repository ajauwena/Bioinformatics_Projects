#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./read_group_adder.sh /scratch/ajauwena/binf_6110/p3/sorted_bam_files /scratch/ajauwena/binf_6110/p3/sorted_bam_files_read_groups

# Command line arguments:
    # $0: ./read_group_adder.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/sorted_bam_files (a directory containing the sorted .bam files for the 10 individual fish).
    # $2: /scratch/ajauwena/binf_6110/p3/sorted_bam_files_read_groups (a directory containing the sorted .bam files for the 10 individual fish with read groups).

# Set a counter.
c=1

# Loop through each sorted .bam file in the appropriate directory.
for file in $1/*
do

    # Extract the file's base name.
    base_name=$(echo ${file} | rev | cut -d '.' -f2 | cut -d '/' -f1 | rev)

    # Inform the user of the file that is being processed.
    echo "Adding a read group to" ${base_name} "..."

    # Replace the read groups in the file.
    java -jar $EBROOTPICARD/picard.jar AddOrReplaceReadGroups I=${file} O=$2/${base_name}_rg.bam RGID=${c} RGLB=lib1 RGPL=illumina RGPU=unit1 RGSM=${base_name}

    # Increment the counter by 1.
    c=$((c + 1))

done