#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=12:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Gunzipping_the_Barley_Reference_Genome
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

dir_barley_reference_genome="/home/ajauwena/scratch/binf_6999/barley_reference_genome"

cp ${dir_barley_reference_genome}/GCA_904849725.1.fasta.gz ${dir_barley_reference_genome}/copy_GCA_904849725.1.fasta.gz

gunzip ${dir_barley_reference_genome}/copy_GCA_904849725.1.fasta.gz

mv ${dir_barley_reference_genome}/copy_GCA_904849725.1.fasta ${dir_barley_reference_genome}/GCA_904849725.1.fasta