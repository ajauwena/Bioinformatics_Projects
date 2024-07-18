#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./snp_number_finder.sh /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered.recode.vcf /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered.recode.vcf /scratch/ajauwena/binf_6110/p3/analyses

# Command line arguments:
    # $0: ./snp_number_finder.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered.recode.vcf (a .vcf file containing variants identified using bcftools mpileup, filtered for quality greater than 30).
    # $2: /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered.recode.vcf (a .vcf file containing variants identified using freebayes, filtered for quality greater than 30).
    # $3: /scratch/ajauwena/binf_6110/p3/analyses (a directory containing files used for analyses).

# Print column headers and write them to a .txt file.
printf "bcftools\tfreebayes\n" > $3/snp_number.txt

# Extract the number of SNPs obtained from calling variants using bcftools mpileup.
snps_bcftools=$(echo ${1} | bcftools stats ${1} | grep "number of SNPs:" | cut -d$'\t' -f4)

# Extract the number of SNPs obtained from calling variants using freebayes.
snps_freebayes=$(echo ${2} | bcftools stats ${2} | grep "number of SNPs:" | cut -d$'\t' -f4)

# Print the numbers of SNPs obtained above and append them to the .txt file.
echo -e "${snps_bcftools}\t${snps_freebayes}" >> $3/snp_number.txt