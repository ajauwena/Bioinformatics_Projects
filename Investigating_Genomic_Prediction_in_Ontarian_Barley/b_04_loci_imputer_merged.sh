#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Imputing_Loci_in_Merged_VCF
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Note: I manually removed the window "CHROM=ENA|CAJHDD010000002|CAJHDD010000002.1, POS=47765" and "CHROM=ENA|CAJHDD010000027|CAJHDD010000027.1, POS=51378" because they only have one position, which prevented Beagle from calculating the appropriate statistical measures needed for imputation.

module load beagle/5.4

dir="/home/ajauwena/scratch/binf_6999/barley_merged"

java -jar ${EBROOTBEAGLE}/beagle.22Jul22.46e.jar gt=$dir/barley_merged_platypus_loci.vcf out=$dir/barley_merged_platypus_loci_imputed