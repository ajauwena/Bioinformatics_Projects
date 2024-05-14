#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=01:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=VCF_to_TAB_May_2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load vcftools/0.1.16

dir="/home/ajauwena/scratch/binf_6999/barley_may_2022/analysis/platypus_barley_may_2022"

vcftools --vcf $dir/barley_may_2022_platypus.vcf --remove-indels --remove-filtered-all --recode --recode-INFO-all --out $dir/barley_may_2022_platypus_snp

cat $dir/barley_may_2022_platypus_snp.recode.vcf | vcf-to-tab > $dir/barley_may_2022_platypus_snp.tab