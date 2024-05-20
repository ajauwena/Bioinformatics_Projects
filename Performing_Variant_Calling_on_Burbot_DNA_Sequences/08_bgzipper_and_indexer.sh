#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./bgzipper_and_indexer.sh /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered.recode.vcf /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered.recode.vcf /scratch/ajauwena/binf_6110/p3/results_bcftools /scratch/ajauwena/binf_6110/p3/results_freebayes

# Command line arguments:
    # $0: ./bgzipper_and_indexer.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered.recode.vcf (a .vcf file containing variants identified using bcftools mpileup, filtered for quality greater than 30).
    # $2: /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered.recode.vcf (a .vcf file containing variants identified using freebayes, filtered for quality greater than 30).
    # $3: /scratch/ajauwena/binf_6110/p3/results_bcftools (a directory containing the results obtained from calling variants using bcftools mpileup).
    # $4: /scratch/ajauwena/binf_6110/p3/results_freebayes (a directory containing the results obtained from calling variants using freebayes).

# Make a copy of the .vcf file obtained from calling variants using bcftools mpileup for bgzipping.
cp ${1} $3/results_bcftools_filtered_for_bgzipping.recode.vcf

# bgzip the .vcf file copy.
bgzip $3/results_bcftools_filtered_for_bgzipping.recode.vcf

# Index the bgzipped file.
bcftools index $3/*.gz

# Make a copy of the .vcf file obtained from calling variants using freebayes for bgzipping.
cp ${2} $4/results_freebayes_filtered_for_bgzipping.recode.vcf

# bgzip the .vcf file copy.
bgzip $4/results_freebayes_filtered_for_bgzipping.recode.vcf

# Index the bgzipped file.
bcftools index $4/*.gz