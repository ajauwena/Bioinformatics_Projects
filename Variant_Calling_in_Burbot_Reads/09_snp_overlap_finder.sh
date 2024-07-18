#!/bin/sh
#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=30000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, run:
    # ./snp_overlap_finder.sh /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered_for_bgzipping.recode.vcf.gz /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered_for_bgzipping.recode.vcf.gz /scratch/ajauwena/binf_6110/p3/analyses

# Command line arguments:
    # $0: ./snp_overlap_finder.sh (this script).
    # $1: /scratch/ajauwena/binf_6110/p3/results_bcftools/results_bcftools_filtered_for_bgzipping.recode.vcf.gz (a bgzipped .vcf file containing variants identified using BCFtools, filtered for quality greater than 30).
    # $2: /scratch/ajauwena/binf_6110/p3/results_freebayes/results_freebayes_filtered_for_bgzipping.recode.vcf.gz (a bgzipped .vcf file containing variants identified using Freebayes, filtered for quality greater than 30).
    # $3: /scratch/ajauwena/binf_6110/p3/analyses (a directory containing files used for analyses).

# Print column headers and write them to a .txt file.
printf "percentage_snp_bcftools\tpercentage_snp_overlap_over_snp_bcftools\tpercentage_snp_overlap_over_snp_freebayes\tpercentage_snp_freebayes\n" > $3/snp_overlap.txt

# Obtain the SNPs that overlap between the two .vcf files.
snp_overlap=$(vcf-compare ${1} ${2} | grep ^VN | cut -f 2-)

# Obtain the percentage of SNPs identified using BCFtools.
snp_bcftools=$(echo ${snp_overlap} | cut -d '%' -f1 | cut -d '(' -f2)
echo "The percentage of SNPs identified using BCFtools:" ${snp_bcftools}

# Obtain the percentage of SNPs that overlap that were also identified using BCFtools.
snp_overlap_over_snp_bcftools=$(echo ${snp_overlap} | cut -d '%' -f2 | cut -d '(' -f2)
echo "The percentage of SNPs that overlap that were also identified using BCFtools:" ${snp_overlap_over_snp_bcftools}

# Obtain the percentage of SNPs that overlap that were also identified using Freebayes.
snp_overlap_over_snp_freebayes=$(echo ${snp_overlap} | cut -d '%' -f3 | cut -d '(' -f2)
echo "The percentage of SNPs that overlap that were also identified using freebayes:" ${snp_overlap_over_snp_freebayes}

# Obtain the percentage of SNPs identified using Freebayes.
snp_freebayes=$(echo ${snp_overlap} | cut -d '%' -f4 | cut -d '(' -f2)
echo "The percentage of SNPs identified using freebayes:" ${snp_freebayes}

# Print the values obtained above and append them to the .txt file.
printf "${snp_bcftools}\t${snp_overlap_over_snp_bcftools}\t${snp_overlap_over_snp_freebayes}\t${snp_freebayes}\n" >> $3/snp_overlap.txt