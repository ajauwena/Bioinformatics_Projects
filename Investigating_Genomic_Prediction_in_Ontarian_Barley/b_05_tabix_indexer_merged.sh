#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Indexing_the_Merged_VCF_Using_Tabix
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load nixpkgs/16.09 intel/2018.3 tabix/0.2.6

dir="/home/ajauwena/scratch/binf_6999/barley_merged"

# Tabix requires the input VCF file to be compressed using "bgzip."
bgzip -c $dir/barley_merged_platypus_loci_imputed.vcf > $dir/barley_merged_platypus_loci_imputed.vcf.gz

tabix $dir/barley_merged_platypus_loci_imputed.vcf.gz