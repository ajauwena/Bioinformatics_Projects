#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=01:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=VCF_to_TAB_June_9_2023
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load vcftools/0.1.16

dir="/home/ajauwena/scratch/binf_6999/barley_june_9_2023/analysis/platypus_barley_june_9_2023"

vcftools --vcf $dir/barley_june_9_2023_platypus.vcf --remove-indels --remove-filtered-all --recode --recode-INFO-all --out $dir/barley_june_9_2023_platypus_snp

cat $dir/barley_june_9_2023_platypus_snp.recode.vcf | vcf-to-tab > $dir/barley_june_9_2023_platypus_snp.tab