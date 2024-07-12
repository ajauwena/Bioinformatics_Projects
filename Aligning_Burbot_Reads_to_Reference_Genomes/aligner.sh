#!/bin/sh

#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=16000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, type the following command in the terminal:
    # sbatch aligner.sh /scratch/ajauwena/binf_6110/project_2/burbot_raw_data /scratch/ajauwena/binf_6110/project_2/burbot_reference_genome /scratch/ajauwena/binf_6110/project_2/cod_reference_genome /scratch/ajauwena/binf_6110/project_2/burbot_alignment_bwa /scratch/ajauwena/binf_6110/project_2/cod_alignment_bwa /scratch/ajauwena/binf_6110/project_2/burbot_alignment_bowtie2 /scratch/ajauwena/binf_6110/project_2/cod_alignment_bowtie2

# Set a variable storing the working directory.
wd=/scratch/ajauwena/binf_6110/project_2

# Set variables storing the burbot and cod reference genomes (i.e., the .fna files).
burbot_ref_gen=$2/GCA_900302385.1_ASM90030238v1_genomic.fna
cod_ref_gen=$3/GCF_902167405.1_gadMor3.0_genomic.fna

# Build the bwa index for the burbot reference genome.
bwa index -a bwtsw ${burbot_ref_gen}

# Build the bwa index for the cod reference genome.
bwa index -a bwtsw ${cod_ref_gen}

# Build the bowtie2 index for the burbot reference genome.
bowtie2-build ${burbot_ref_gen} $2/GCA_900302385.1_ASM90030238v1_genomic

# Build the bowtie2 index for the cod reference genome.
bowtie2-build ${cod_ref_gen} $3/GCF_902167405.1_gadMor3.0_genomic

# Loop through each raw data file.
for file in $1/*
do

    # Extract the file's base name without the .fq.
    base_name_fq=$(echo ${file} | rev | cut -d '/' -f1 | rev)
    base_name=$(echo ${base_name_fq} | cut -d '.' -f1)

    # Align the file to the burbot reference genome using bwa, then output the results to a .sam file in the appropriate directory.
    bwa mem -t 4 ${burbot_ref_gen} ${file} > $4/${base_name}.sam

    # Align the file to the cod reference genome using bwa, then output the results to a .sam file in the appropriate directory.
    bwa mem -t 4 ${cod_ref_gen} ${file} > $5/${base_name}.sam

    # Align the file to the burbot reference genome using bowtie2, then output the results to a .sam file in the appropriate directory.
    bowtie2 -x $2/GCA_900302385.1_ASM90030238v1_genomic -U ${file} -S $6/${base_name}.sam --very-sensitive-local

    # Align the file to the cod reference genome using bowtie2, then output the results to a .sam file in the appropriate directory.
    bowtie2 -x $3/GCF_902167405.1_gadMor3.0_genomic -U ${file} -S $7/${base_name}.sam --very-sensitive-local

done