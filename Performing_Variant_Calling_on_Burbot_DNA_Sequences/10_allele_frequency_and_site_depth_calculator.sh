#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./allele_frequency_and_site_depth_calculator.sh /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered.recode.vcf /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered.recode.vcf /scratch/ajauwena/binf_6110/p3/analyses

# Command line arguments:
    # $0: ./allele_frequency_and_site_depth_calculator.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered.recode.vcf (a .vcf file containing variants identified using bcftools mpileup, filtered for quality greater than 30).
    # $2: /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered.recode.vcf (a .vcf file containing variants identified using freebayes, filtered for quality greater than 30).
    # $3: /scratch/ajauwena/binf_6110/p3/analyses (a directory containing files used for analyses).

# Extract the base names of the two inputted .vcf files.
base_name_bcftools=$(echo ${1} | rev | cut -d '.' -f3 | cut -d '/' -f1 | rev)
base_name_freebayes=$(echo ${2} | rev | cut -d '.' -f3 | cut -d '/' -f1 | rev)

# Obtain the minor allele frequency for each .vcf file.
vcftools --vcf ${1} --freq2 --out $3/${base_name_bcftools}_minor_allele_frequency
vcftools --vcf ${2} --freq2 --out $3/${base_name_freebayes}_minor_allele_frequency
# The output files will have the suffix ".frq" (VCFtools, n.d.).

# Obtain the depth of all snps, summed across individuals, for each .vcf file.
vcftools --vcf ${1} --site-depth --out $3/${base_name_bcftools}_summed_depth
vcftools --vcf ${2} --site-depth --out $3/${base_name_freebayes}_summed_depth
# The output files will have the suffix ".ldepth" (VCFtools, n.d.).

# Obtain the mean depth for each .vcf file, measured in reads per locus per individual.
vcftools --vcf ${1} --site-mean-depth --out $3/${base_name_bcftools}_mean_depth
vcftools --vcf ${2} --site-mean-depth --out $3/${base_name_freebayes}_mean_depth
# The output files will have the suffix ".ldepth.mean" (VCFtools, n.d.).