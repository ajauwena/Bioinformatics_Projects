#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script on the directory "results_bcftools," run:
    # ./filterer.sh /scratch/ajauwena/binf_6110/p3/results_bcftools

# Command line arguments (when executed on the directory "results_bcftools"):
    # $0: ./filterer.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/results_bcftools (a directory containing the results obtained from calling variants using bcftools mpileup).

# To execute this script on the directory "results_freebayes," run:
    # ./filterer.sh /scratch/ajauwena/binf_6110/p3/results_freebayes

# Command line arguments (when executed on the directory "results_freebayes"):
    # $0: ./filterer.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/results_freebayes (a directory containing the results obtained from calling variants using freebayes).

# Loop through each file in the appropriate directory.
for file in $1/*
do

    # Extract the file's extension name.
    extension_name=$(echo ${file} | rev | cut -d '.' -f1 | rev)

    # If the file's extension name is "vcf"...
    if [ ${extension_name} == "vcf" ]
    then

        # Extract the file's base name.
        base_name=$(echo ${file} | rev | cut -d '.' -f2 | cut -d '/' -f1 | rev)

        # Filter the file using reasonable thresholds.
        vcftools --vcf ${file} --minQ 30 --mac 1 --max-maf 0.5 --max-missing 1 --remove-indels --max-alleles 2 --recode --out $1/${base_name}_filtered # (VCFtools, n.d.).

    fi

done