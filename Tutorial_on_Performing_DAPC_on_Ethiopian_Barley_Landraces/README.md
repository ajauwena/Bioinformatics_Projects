# Tutorial on Performing DAPC on Ethiopian Barley Landraces

## Overview

In recent years, "Discriminant Analysis of Principal Components" ("DAPC") has become the method of choice for inferring genetic structure and identifying alleles responsible for phenotypic differentiation within populations.<sup>3,7</sup> It has been widely applied in genetic analyses; for example, it has been used to assess diversity and structure in potatoes using SNP markers,<sup>2</sup> analyzing genetic diversity and population structure in sweet cherries to develop more resilient cultivars,<sup>1</sup> and evaluating genetic diversity and population structure in watermelons using SNP data obtained via genotyping by sequencing (GBS).<sup>6</sup> The popularity of DAPC stems from the fact that it offers the benefits of both principal component analysis (PCA) and discriminant analysis (DA) while minimizing their drawbacks.<sup>5</sup>

This project serves as a tutorial on performing DAPC in R to identify genetic groups in a dataset of 384 Ethiopian barley landrace genotypes from Dido et al. (2022). The original analysis by Dido et al. (2021) aimed to investigate genetic variation in Ethiopian barley using SSR markers, but did not provide detailed steps. The goal of this tutorial is to replicate their findings. The dataset for this project can be downloaded [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.4tmpg4f8v) in XLSX format.<sup>4</sup>

## Files

This repository contains three kinds of files.

### 1. Tutorial

#### <em>tutorial.pdf</em>

Describes how to perform DAPC in R to identify genetic groups in a dataset of Ethiopian barley landrace genotypes.

### 2. Dataset

#### <em>ethiopian_barley_dataset.xlsx</em>

Contains genotype data from 384 Ethiopian barley landraces.

### 3. R Script

#### <em>dapc_performer.R</em>

Performs DAPC on a dataset of Ethiopian barley landrace genotypes and visualizes the results.

## References

1. Campoy, J. A., Lerigoleur-Balsemin, E., Christmann, H., Beauvieux, R., Girollet, N., Quero-Garcia, J., Dirlewanger, E., & Barreneche, T. (2016). Genetic diversity, linkage disequilibrium, population structure and construction of a core collection of Prunus avium L. landraces and bred cultivars. <em>BMC Plant Biology, 16</em>(1). https://doi.org/10.1186/s12870-016-0712-9
2. Deperi, S. I., Tagliotti, M. E., Bedogni, M. C., Manrique-Carpintero, N. C., Coombs, J. E., Zhang, R., Douches, D. S., & Huarte, M. (2018). Discriminant analysis of principal components and pedigree assessment of genetic diversity and population structure in a tetraploid potato panel using SNPs. <em>PLOS ONE, 13</em>(3), e0194398. https://doi.org/10.1371/journal.pone.0194398
3. Dido, A. A., Degefu, D. T., Assefa, E., Krishna, M. S. R., Singh, B. P., & Tesfaye, K. (2021). Spatial and temporal genetic variation in Ethiopian barley (Hordeum vulgare L.) landraces as revealed by simple sequence repeat (SSR) markers. <em>Agriculture & Food Security, 10</em>(1). https://doi.org/10.1186/s40066-021-00336-3
4. Dido, A. A., Singh, B. J. K., Assefa, E., Krishna, M. S. R., Degefu, D., & Tesfaye, K. (2022). Spatial and temporal genetic variation in Ethiopian barley (Hordeum vulgare L.) landraces as revealed by simple sequence repeat (SSR) markers. <em>Dryad</em>. https://doi.org/10.5061/dryad.4tmpg4f8v
5. Jombart, T., Devillard, S., & Balloux, F. (2010). Discriminant analysis of principal components: a new method for the analysis of genetically structured populations. <em>BMC Genetics, 11</em>(1), 94. https://doi.org/10.1186/1471-2156-11-94
6. Lee, K. J., Lee, J., Sebastin, R., Shin, M., Kim, S., Cho, G., & Hyun, D. Y. (2019). Genetic Diversity Assessed by Genotyping by Sequencing (GBS) in Watermelon Germplasm. <em>Genes, 10</em>(10), 822. https://doi.org/10.3390/genes10100822
7. Miller, J. D., Cullingham, C. I., & Peery, R. M. (2020). The influence of a priori grouping on inference of genetic clusters: simulation study and literature review of the DAPC method. <em>Heredity, 125</em>(5), 269–280. https://doi.org/10.1038/s41437-020-0348-2