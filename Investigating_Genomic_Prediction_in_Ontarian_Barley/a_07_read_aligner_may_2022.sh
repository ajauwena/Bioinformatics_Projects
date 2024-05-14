#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Aligning_Reads_to_the_Reference_Genome_May_2022
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load bwa
module load samtools

dir="/home/ajauwena/scratch/binf_6999"

# Align reads using "bwa mem," then sort the aligned reads as BAM files using "samtools sort."
parallel -j 4 bwa mem -t 4 ${dir}/barley_reference_genome/GCA_904849725.1.fasta.gz \
${dir}/barley_may_2022/trimmed_twice/{}_trim_2x_forward.fastq.gz \
${dir}/barley_may_2022/trimmed_twice/{}_trim_2x_reverse.fastq.gz "|" \
samtools sort -o ${dir}/barley_may_2022/bam/{}_sort.bam ::: $(ls -1 ${dir}/barley_may_2022/trimmed_twice/*.fastq.gz | xargs -n 1 basename | sed 's/_.*//' | sort -u)

# Create both index (".bai") and column-store index (".bam.csi") files for the sorted BAM files using "samtools index."
parallel -j 16 samtools index -c ${dir}/barley_may_2022/bam/{}_sort.bam ${dir}/barley_may_2022/bam/{}_sort.bam.csi ::: $(ls -1 ${dir}/barley_may_2022/bam/*_sort.bam | xargs -n 1 basename | sed 's/_.*//')