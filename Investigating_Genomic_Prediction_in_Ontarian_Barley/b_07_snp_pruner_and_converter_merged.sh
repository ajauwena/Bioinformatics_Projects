#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Pruning_SNPs_and_Converting_Them_to_the_"012"_Format_in_Merged_VCF
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load vcftools/0.1.16

dir="/home/ajauwena/scratch/binf_6999/barley_merged"

# Filter the VCF file by the SNP IDs that survived pruning (see "code_pca_imputed_data_draft_1.R").
vcftools --vcf $dir/barley_merged_platypus_loci_imputed_subsetted.vcf --snps $dir/pruned_snp_ids.txt --recode --recode-INFO-all --out $dir/barley_merged_platypus_loci_imputed_pruned_subsetted

vcftools --vcf $dir/barley_merged_platypus_loci_imputed_pruned_subsetted.recode.vcf --012 --out $dir/barley_merged_platypus_loci_imputed_pruned_subsetted_012
