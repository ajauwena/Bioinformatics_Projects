#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Imputing_Loci_May_2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load beagle/5.4

dir="/home/ajauwena/scratch/binf_6999/barley_may_2022/analysis/tassel_barley_may_2022"

java -jar ${EBROOTBEAGLE}/beagle.22Jul22.46e.jar gt=$dir/barley_may_2022_platypus_loci_filtered_filtered.vcf out=$dir/barley_may_2022_platypus_loci_filtered_filtered_imputed