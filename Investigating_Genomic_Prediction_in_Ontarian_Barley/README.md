
# Investigating Genomic Prediction in Ontarian Barley Varieties

This project has three aims, which are as follows:
1. To elucidate the historical yield trend of barley varieties in Ontario, Canada and determine whether this trend is primarily influenced by genetic or environmental factors.
2. To identify genome-wide allelic differences in these barley varieties and determine whether they correlate with yield differences over time.
3. To investigate whether genomic prediction can estimate the performance of an unknown barley variety using the identified genome-wide allelic differences.

## Files

This repository contains four kinds of files.

### Bash (.sh) Scripts

The Bash scripts are divided into two categories: those named "a_*.sh" and those named "b_*.sh".
1. "a_*.sh" Files – Files that serve to execute the "Fast-GBS" pipeline proposed by Torkamaneh et al. (2017).
2. "b_*.sh" Files – Files that serve to process called single nucleotide polymorphisms (SNPs) and convert them into a format suitable for genomic prediction.

### R (.R) Scripts

There are three R scripts included in this repository.
1.  "historical_yield_trend_analyzer.R" – A script that elucidates the historical yield trends of 372 barley varieties in Ontario, Canada from 1958 to 2021.
2.  "pca_performer.R" – A script that performs principal components analysis (PCA) on 54 barley varieties with both SNP and yield data. This script aims to identify genome-wide allelic differences in these barley varieites and determine whether they correlate with yield differences over time..
3.  "genomic_prediction_performer.R" – A script that investigates the utility of genomic prediction on estimating an unknown variety's performance using the genome-wide allelic differences identified in the "pca_performer.R" script.

### Final Report (.pdf)

The final report, named "report.pdf", outlines all the project's methods, results, and implications of those results.

### Project Workflow (.txt)

The project workflow will be added to this repository. It will provide a detailed, step-by-step overview of the project's methods. For each step, it will also specify the tools used, the scripts used, the input files required, and the output files produced.