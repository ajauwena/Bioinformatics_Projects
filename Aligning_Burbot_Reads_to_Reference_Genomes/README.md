# Aligning Burbot Reads to Reference Genomes

This project aims to investigate how the choice of reference genome and alignment software impacts on read alignment success.

## Files

This repository contains four kinds of files.

### Bash Scripts

There are two Bash scripts included in this repository.
1.  "aligner.sh" – A script that aligns 10 burbot reads to burbot and cod reference genomes using the "Burrows-Wheeler Aligner" ("BWA") and "Bowtie 2" software.
2.  "alignment_statistics_calculator.sh" – A script that outputs several alignment statistics of interest into .txt files. These statistics include the i) number of reads mapped, ii) raw total sequences, iii) error rate, iv) average length, v) average quality, and vi) alignment rate.

### Python Scripts
There are four Python scripts included in this repository.
1. "alignment_statistics_csv_burbot_bwa.py" – A script that outputs the alignment statistics obtained by aligning burbot reads to the burbot reference genome using BWA to CSV files.
2. "alignment_statistics_csv_cod_bwa.py" – A script that outputs the alignment statistics obtained by aligning burbot reads to the cod reference genome using BWA to CSV files.
3. "alignment_statistics_csv_burbot_bowtie2.py" – A script that outputs the alignment statistics obtained by aligning burbot reads to the burbot reference genome using Bowtie 2 to CSV files.
4. "alignment_statistics_csv_cod_bowtie2.py" – A script that outputs the alignment statistics obtained by aligning burbot reads to the cod reference genome using Bowtie 2 to CSV files.

### R Scripts

The "alignment_statistics_grapher.R" script visualizes the the alignment rates of the 10 burbot reads when aligned to burbot and cod reference genomes using BWA and Bowtie 2. It also performs t-tests to see whether there is a significant difference in the mean alignment when using the burbot vs. cod reference genomes, as well as when using BWA vs. Bowtie 2.

### Report

The report, named "report.pdf", outlines the project's background, methods, results, and implications of those results.