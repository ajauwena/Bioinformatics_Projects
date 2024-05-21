# Burbot Read Alignment to Reference Genomes

## Overview

High-throughput sequencing has generated vast quantities of DNA and RNA reads from various organisms.<sup>2</sup> As a result, researchers are faced with the task of identifying the genomic regions to which these reads belong.<sup>2</sup> The most common approach for this task is to align reads to a reference genome, which helps in determining sequence variations that influence traits within a population, as well as deducing phylogenetic relationships.<sup>2,4</sup>

Unfortunately, this process is computationally demanding and requires efficient algorithms, especially when involving genomes that are large and/or contain long and complex repeat patterns.<sup>1,2,3</sup> Additionally, researchers must choose an appropriate reference genome: either a complete genome from a distantly related species or a fragmented genome from the same species. Each choice has its own set of challenges.

This project aims to explore how the choice of reference genome and alignment software affects alignment success. By aligning 10 burbot (<em>Lota lota</em>) reads to both a fragmented burbot genome and a complete cod genome using the "Burrows-Wheeler Aligner" ("BWA") and "Bowtie 2" alignment software, this project can hopefully identify the most effective alignment method for maximizing alignment success rates.

## Files

This repository contains four kinds of files.

### 1. Bash Scripts

#### <em>aligner.sh</em>

Aligns 10 burbot reads to burbot and cod reference genomes using BWA and Bowtie 2.

#### <em>alignment_statistics_calculator.sh</em>

Outputs several alignment statistics of interest into .txt files. These statistics include the i) number of reads mapped, ii) raw total sequences, iii) error rate, iv) average length, v) average quality, and vi) alignment rate.

### 2. Python Scripts

#### <em>alignment_statistics_csv_burbot_bwa.py</em>

Outputs the alignment statistics obtained by aligning burbot reads to the burbot reference genome using BWA to CSV files.

#### <em>alignment_statistics_csv_cod_bwa.py</em>

Outputs the alignment statistics obtained by aligning burbot reads to the cod reference genome using BWA to CSV files.

#### <em>alignment_statistics_csv_burbot_bowtie2.py</em>

Outputs the alignment statistics obtained by aligning burbot reads to the burbot reference genome using Bowtie 2 to CSV files.

#### <em>alignment_statistics_csv_cod_bowtie2.py</em>

Outputs the alignment statistics obtained by aligning burbot reads to the cod reference genome using Bowtie 2 to CSV files.

### 3. R Script

#### <em>alignment_statistics_grapher.R</em>

Visualizes the the alignment rates of the 10 burbot reads when aligned to burbot and cod reference genomes using BWA and Bowtie 2. It also performs t-tests to see whether there is a significant difference in the mean alignment when using the burbot versus cod reference genomes, as well as when using BWA versus Bowtie 2.

### 4. Report

#### <em>report.pdf</em>

Outlines the project's background, methods, results, and implications of those results.

## References

1. Phan, V., Gao, S., Tran, Q., & Vo, N. S. (2015). How genome complexity can explain the difficulty of aligning reads to genomes. <em>BMC Bioinformatics, 16</em>(S3). https://doi.org/10.1186/1471-2105-16-S17-S3
2. Reinert, K., Langmead, B., Weese, D., & Evers, D. J. (2015). Alignment of Next-Generation Sequencing Reads. <em>Annual Reviews of Genomics and Human Genetics, 16</em>, 133-151. https://doi.org/10.1146/annurev-genom-090413-025358
3. Trapnell, C., & Salzberg, S. L. (2009). How to map billions of short reads onto genomes. <em>Nature Biotechnology, 27</em>, 455-457. https://doi.org/10.1038/nbt0509-455
4. Valiente-Mullor, C., Beamud, B., Ansari, I., Francés-Cuesta, C., García-González, N., Mejía, L., Ruiz-Hueso, P., & González-Candelas, F. (2021). One is not enough: On the effects of reference genome for the mapping and subsequent analyses of short-reads. <em>PLOS Computational Biology, 17</em>(1). https://doi.org/10.1371/journal.pcbi.1008678