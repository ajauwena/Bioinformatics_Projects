#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=12:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Variant_Calling_May_2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load nixpkgs/16.09 gcc/7.3.0 platypus/0.8.1

dir="/home/ajauwena/scratch/binf_6999"

Platypus.py callVariants --bamFiles=$dir/barley_may_2022/bam/bam_list_may_2022.txt --refFile=$dir/barley_reference_genome/GCA_904849725.1.fasta --output=$dir/barley_may_2022/analysis/platypus_barley_may_2022/barley_may_2022_platypus.vcf \
	--logFileName=$dir/barley_may_2022/analysis/platypus_barley_may_2022/barley_may_2022_platypus_log.txt --nCPU=8 --minMapQual=20 \
	--minBaseQual=20 --minGoodQualBases=5 --badReadsThreshold=10 --abThreshold=0.01 \
	--minReads=2 --genIndels=0 --sbThreshold=0.01 \