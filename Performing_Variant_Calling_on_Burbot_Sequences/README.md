# Performing Variant Calling on Burbot Sequences

## Overview

Variant calling aims to identify variations in DNA sequences using high-throughput sequencing (HTS) technology, distinguishing true variants from experimental errors.<sup>2,9</sup> Clinical genomic studies typically use variant calling to identify single nucleotide variants (SNVs), small insertions and deletions (indels), and structural variants (SVs).<sup>2,8,11</sup> Variant calling supports genomic research in many ways, such as by aiding in fine mapping and in understanding the genomic bases of diseases.<sup>6,8,9</sup>

BCFtools and Freebayes are popular variant calling software.<sup>2</sup> BCFtools is a Hidden Markov model-based variant caller that uses the Hardy-Weinberg equilibrium to evaluate the most likely genotype for each position in a genome.<sup>3</sup> It is speedy and memory-efficient, but struggles with very large files and has limitations with certain organism types and large genomes.<sup>3,5</sup> On the other hand, Freebayes is a Bayesian variant caller designed to identify small variations by literally aligning sequences to a target.<sup>1,4</sup> Although it performs well with various genome sequences, it requires extensive processing and filtering to avoid issues like having low specificity and sensitivity or failing to detect more SNPs with increasing input read depths.<sup>5,7,10</sup>

This project aims to compare the performance of BCFtools and Freebayes in identifying variants within 10 burbot (<em>Lota lota</em>) sequences.

## Files

This repository contains three kinds of files.

### 1. Bash Scripts

#### <em>01_sam_to_sorted_bam_converter.sh</em>

Converts the 10 sequences from SAM to BAM format and sorts them.

#### <em>02_variant_caller_and_vcf_file_combiner_bcftools.sh</em>

Calls variants using BCFtools and outputs the results to a VCF file.

#### <em>03_read_group_adder.sh</em>

Adds read groups to the 10 sequences in BAM format to prepare them for variant calling using Freebayes (see <em>05_variant_caller_and_vcf_file_combiner_freebayes.sh</em>).

#### <em>04_read_group_reindexer.sh</em>

Reindexes the read groups for the 10 sequences in BAM format to prepare them for variant calling using Freebayes (see <em>05_variant_caller_and_vcf_file_combiner_freebayes.sh</em>).

#### <em>05_variant_caller_and_vcf_file_combiner_freebayes.sh</em>

Calls variants using Freebayes and outputs the results to a VCF file.

#### <em>06_filterer.sh</em>

Filters the two VCF files resulting from variant calling using BCFtools and Freebayes (hereinafter, "two VCF files") using appropriate thresholds.

#### <em>07_snp_number_finder.sh</em>

Extracts the number of single nucleotide polymorphisms (SNPs) obtained from variant calling using BCFtools and Freebayes.

#### <em>08_bgzipper_and_indexer.sh</em>

Zips the two VCF files and indexes them.

#### <em>09_snp_overlap_finder.sh</em>

Obtains the percentage of SNPs identified by BCFtools and Freebayes.

#### <em>10_allele_frequency_and_site_depth_calculator.sh</em>

Obtains the minor allele frequency, summed depth, and mean depth for the two VCF files.

### 2. R Script

#### <em>11_result_analyzer.R</em>

Compares the performance of BCFtools and Freebayes in identifying variants.

### 3. Report

#### <em>report.pdf</em>

Outlines the project's background, methods, results, and implications of those results.

## References

1. Bian, X., Zhu, B., Wang, M., Hu, Y., Chen, Q., Nguyen, C., Hicks, B., & Meerzaman, D. (2018). Comparing the performance of selected variant callers using synthetic data and genome segmentation. <em>BMC Bioinformatics, 19</em>(1). https://doi.org/10.1186/s12859-018-2440-7
2. Bohannan, Z. S., & Mitrofanova, A. (2019). Calling Variants in the Clinic: Informed Variant Calling Decisions Based on Biological, Clinical, and Laboratory Variables. <em>Computational and Structural Biotechnology Journal, 17</em>, 561–569. https://doi.org/10.1016/j.csbj.2019.04.002
3. Danecek, P., Bonfield, J. K., Liddle, J., Marshall, J., Ohan, V., Pollard, M. O., Whitwham, A., Keane, T. E., McCarthy, S. A., Davies, R. J. O., & Li, H. (2021). Twelve years of SAMtools and BCFtools. <em>GigaScience, 10</em>(2). https://doi.org/10.1093/gigascience/giab008
4. <em>freebayes</em>, a haplotype-based variant detector. (n.d.). GitHub. Retrieved March 21, 2023, from https://github.com/freebayes/freebayes
5. Liu, J., Shen, Q., & Bao, H. (2022). Comparison of seven SNP calling pipelines for the next-generation sequencing data of chickens. <em>PLOS ONE, 17</em>(1), e0262574. https://doi.org/10.1371/journal.pone.0262574
6. Schaid, D. J., Chen, W., & Larson, N. B. (2018). From genome-wide associations to candidate causal variants by statistical fine-mapping. <em>Nature Reviews Genetics, 19</em>(8), 491–504. https://doi.org/10.1038/s41576-018-0016-z
7. Stegemiller, M. R., Redden, R. R., Notter, D. R., Taylor, T., Taylor, J. B., Cockett, N. E., Heaton, M. P., Kalbfleisch, T. S., & Murdoch, B. M. (2023). Using whole genome sequence to compare variant callers and breed differences of US sheep. <em>Frontiers in Genetics, 13</em>. https://doi.org/10.3389/fgene.2022.1060882
8. <em>Summoning insights: NGS variant calling best practices</em>. (2023, January 11). OGT. Retrieved March 16, 2023, from https://www.ogt.com/ca/about-us/ogt-blog/summoning-insights-ngs-variant-calling-best-practices/
9. <em>Variant Calling</em>. (n.d.). CD Genomics. Retrieved March 15, 2023, from https://www.cd-genomics.com/variant-calling.html
10. Yao, Z., You, F. M., N’Diaye, A. T., Majidi, M. M., McCartney, C. A., Hiebert, C. W., Pozniak, C. J., & Xu, W. (2020). Evaluation of variant calling tools for large plant genome re-sequencing. <em>BMC Bioinformatics, 21</em>(1). https://doi.org/10.1186/s12859-020-03704-1
11. Zverinova, S., & Guryev, V. (2021). Variant calling: Considerations, practices, and developments. <em>Human Mutation, 43</em>(8), 976–985. https://doi.org/10.1002/humu.24311