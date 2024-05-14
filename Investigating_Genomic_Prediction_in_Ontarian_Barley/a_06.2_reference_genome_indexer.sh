#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=6:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Indexing_the_Barley_Reference_Genome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load bwa samtools bowtie2

dir_barley_reference_genome="/home/ajauwena/scratch/binf_6999/barley_reference_genome"

samtools faidx ${dir_barley_reference_genome}/GCA_904849725.1.fasta

bwa index -p ${dir_barley_reference_genome}/GCA_904849725.1.fasta.gz -a bwtsw ${dir_barley_reference_genome}/GCA_904849725.1.fasta.gz
