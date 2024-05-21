#!/bin/sh

#SBATCH --account=def-lukens
#SBATCH --time=0-10:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=16000 # requested memory (in MB)
#SBATCH --mail-type=END

# To execute this script, type the following command in the terminal:
    # sbatch alignment_statistics_calculator.sh /scratch/ajauwena/binf_6110/project_2/burbot_alignment_bwa /scratch/ajauwena/binf_6110/project_2/cod_alignment_bwa /scratch/ajauwena/binf_6110/project_2/burbot_alignment_bowtie2 /scratch/ajauwena/binf_6110/project_2/cod_alignment_bowtie2

# Set a variable storing the working directory.
wd=/scratch/ajauwena/binf_6110/project_2

# Loop through each .sam file in the directory "burbot_alignment_bwa".
for file in $1/*
do

    # Extract the file's base name without the .fq.
    base_name=$(echo ${file} | cut -d '/' -f 7 | cut -d '.' -f 1)

    # Obtain the number of reads mapped.
    reads_mapped=$(samtools stats ${file} | grep 'reads mapped:' | awk '{print $4}')

    # Obtain the number of raw total sequences.
    raw_total_sequences=$(samtools stats ${file} | grep 'raw total sequences:' | awk '{print $5}')

    # Obtain the error rate.
    error_rate=$(samtools stats ${file} | grep 'error rate:' | awk '{print $4}')

    # Obtain the average length.
    average_length=$(samtools stats ${file} | grep 'average length:' | awk '{print $4}')

    # Obtain the average quality.
    average_quality=$(samtools stats ${file} | grep 'average quality:' | awk '{print $4}')

    # Calculate the alignment completeness by dividing the reads mapped by the raw total sequences.
    alignment_completeness=$(echo "scale=3; ${reads_mapped}/${raw_total_sequences}" | bc)

    # Output all the results into a .txt file in the appropriate directory.
    echo "Reads_Mapped: ${reads_mapped} Raw_Total_Sequences: ${raw_total_sequences} Error_Rate: ${error_rate} Average_Length: ${average_length} Average_Quality: ${average_quality} Alignment_Completeness: ${alignment_completeness}" > ${wd}/alignment_statistics/burbot_statistics_bwa/${base_name}.txt

done

# Loop through each .sam file in the directory "cod_alignment_bwa".
for file in $2/*
do

    # Extract the file's base name without the .fq.
    base_name=$(echo ${file} | cut -d '/' -f 7 | cut -d '.' -f 1)

    # Obtain the number of reads mapped.
    reads_mapped=$(samtools stats ${file} | grep 'reads mapped:' | awk '{print $4}')

    # Obtain the number of raw total sequences.
    raw_total_sequences=$(samtools stats ${file} | grep 'raw total sequences:' | awk '{print $5}')

    # Obtain the error rate.
    error_rate=$(samtools stats ${file} | grep 'error rate:' | awk '{print $4}')

    # Obtain the average length.
    average_length=$(samtools stats ${file} | grep 'average length:' | awk '{print $4}')

    # Obtain the average quality.
    average_quality=$(samtools stats ${file} | grep 'average quality:' | awk '{print $4}')

    # Calculate the alignment completeness by dividing the reads mapped by the raw total sequences.
    alignment_completeness=$(echo "scale=3; ${reads_mapped}/${raw_total_sequences}" | bc)

    # Output all the results into a .txt file in the appropriate directory.
    echo "Reads_Mapped: ${reads_mapped} Raw_Total_Sequences: ${raw_total_sequences} Error_Rate: ${error_rate} Average_Length: ${average_length} Average_Quality: ${average_quality} Alignment_Completeness: ${alignment_completeness}" > ${wd}/alignment_statistics/cod_statistics_bwa/${base_name}.txt

done

# Loop through each .sam file in the directory "burbot_alignment_bowtie2".
for file in $3/*
do

    # Extract the file's base name without the .fq.
    base_name=$(echo ${file} | cut -d '/' -f 7 | cut -d '.' -f 1)

    # Obtain the number of reads mapped.
    reads_mapped=$(samtools stats ${file} | grep 'reads mapped:' | awk '{print $4}')

    # Obtain the number of raw total sequences.
    raw_total_sequences=$(samtools stats ${file} | grep 'raw total sequences:' | awk '{print $5}')

    # Obtain the error rate.
    error_rate=$(samtools stats ${file} | grep 'error rate:' | awk '{print $4}')

    # Obtain the average length.
    average_length=$(samtools stats ${file} | grep 'average length:' | awk '{print $4}')

    # Obtain the average quality.
    average_quality=$(samtools stats ${file} | grep 'average quality:' | awk '{print $4}')

    # Calculate the alignment completeness by dividing the reads mapped by the raw total sequences.
    alignment_completeness=$(echo "scale=3; ${reads_mapped}/${raw_total_sequences}" | bc)

    # Output all the results into a .txt file in the appropriate directory.
    echo "Reads_Mapped: ${reads_mapped} Raw_Total_Sequences: ${raw_total_sequences} Error_Rate: ${error_rate} Average_Length: ${average_length} Average_Quality: ${average_quality} Alignment_Completeness: ${alignment_completeness}" > ${wd}/alignment_statistics/burbot_statistics_bowtie2/${base_name}.txt

done

# Loop through each .sam file in the directory "cod_alignment_bowtie2".
for file in $4/*
do

    # Extract the file's base name without the .fq.
    base_name=$(echo ${file} | cut -d '/' -f 7 | cut -d '.' -f 1)

    # Obtain the number of reads mapped.
    reads_mapped=$(samtools stats ${file} | grep 'reads mapped:' | awk '{print $4}')

    # Obtain the number of raw total sequences.
    raw_total_sequences=$(samtools stats ${file} | grep 'raw total sequences:' | awk '{print $5}')

    # Obtain the error rate.
    error_rate=$(samtools stats ${file} | grep 'error rate:' | awk '{print $4}')

    # Obtain the average length.
    average_length=$(samtools stats ${file} | grep 'average length:' | awk '{print $4}')

    # Obtain the average quality.
    average_quality=$(samtools stats ${file} | grep 'average quality:' | awk '{print $4}')

    # Calculate the alignment completeness by dividing the reads mapped by the raw total sequences.
    alignment_completeness=$(echo "scale=3; ${reads_mapped}/${raw_total_sequences}" | bc)

    # Output all the results into a .txt file in the appropriate directory.
    echo "Reads_Mapped: ${reads_mapped} Raw_Total_Sequences: ${raw_total_sequences} Error_Rate: ${error_rate} Average_Length: ${average_length} Average_Quality: ${average_quality} Alignment_Completeness: ${alignment_completeness}" > ${wd}/alignment_statistics/cod_statistics_bowtie2/${base_name}.txt

done
