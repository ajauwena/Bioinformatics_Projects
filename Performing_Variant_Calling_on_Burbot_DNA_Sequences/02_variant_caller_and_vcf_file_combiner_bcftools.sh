#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./variant_caller_and_vcf_file_combiner_bcftools.sh /scratch/ajauwena/binf_6110/p3/burbot_genome /scratch/ajauwena/binf_6110/p3/sorted_bam_files /scratch/ajauwena/binf_6110/p3/results_bcftools

# Command line arguments:
    # $0: ./variant_caller_and_vcf_file_combiner_bcftools.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/burbot_genome (a directory containing the burbot reference genome).
    # $2: /scratch/ajauwena/binf_6110/p3/sorted_bam_files (a directory containing the sorted .bam files for the 10 individual fish).
    # $3: /scratch/ajauwena/binf_6110/p3/results_bcftools (a directory containing the results obtained from calling variants using bcftools mpileup).

# Loop through each sorted .bam file in the appropriate directory.
for file in $2/*
do

    # Append the file to a list.
    echo ${file} >> $3/bam_file_list_bcftools.txt

done

# Call variants in the files in the list using bcftools mpileup.
bcftools mpileup -a DP,AD -f $1/burbot_2021.fasta -b $3/bam_file_list_bcftools.txt | bcftools call -m --variants-only > $3/results_bcftools.vcf