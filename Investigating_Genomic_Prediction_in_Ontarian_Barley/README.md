
# Investigating Genomic Prediction in Ontarian Barley

## Overview

Barley (<em>Hordeum vulgare</em> L.) is a key cereal crop in Ontario, Canada, and increasing its production could significantly boost food supply in the region.<sup>4</sup> Investigating the historical yields of Ontarian barley varieties can reveal yield trends and help elucidate the influence of genetic and environmental factors on these yields.<sup>6</sup> Here, genetic factors refer to the breeding of genetically superior varieties, while environmental factors encompass factors unrelated to breeding efforts, such as improved farming practices and climatic changes.<sup>5</sup>

Genetic improvement via breeding efforts is an essential component for accelerating food production.<sup>1</sup> Genomic prediction can enhance this process – it involves using statistical models to predict crop performance based on genotype, and it has been successfully applied to other grains like rice and winter wheat.<sup>2,3,6,8</sup>

This project has three objectives, which are:
1. To analyze the historical yield trend of Ontarian barley varieties and determine whether this trend is influenced primarily by genetic or environmental factors.
2. To identify genome-wide allelic differences in these varieties and their correlation with yield changes.
3. To explore the potential of genomic prediction in estimating the performance of unknown varieties based on these identified allelic differences.

## Files

### 1. Bash Scripts

The Bash scripts are divided into two categories: those named "a_\*.sh" and those named "b_\*.sh".

#### <em>a_\*.sh</em>

Execute the Fast-GBS pipeline proposed by Torkamaneh et al. (2017).

#### <em>b_\*.sh</em>

Process called single nucleotide polymorphisms (SNPs) and convert them into a format suitable for genomic prediction.

### 2. R Scripts

#### <em>historical_yield_trend_analyzer.R</em>

This script elucidates the historical yield trends of 372 barley varieties in Ontario, Canada from 1958 to 2021.

#### <em>pca_performer.R</em>

This script performs principal component analysis (PCA) on 54 barley varieties with both SNP and yield data. It aims to identify genome-wide allelic differences in these varieties and determine whether they correlate with yield differences over time.

#### <em>genomic_prediction_performer.R</em>

This script investigates the utility of genomic prediction on estimating an unknown variety's performance using the genome-wide allelic differences identified in the "pca_performer.R" script.

### 3. Report

#### <em>report.pdf</em>

Outlines the project's background, methods, results, and implications of those results.

### 4. Workflow

#### <em>workflow.txt</em>

Outlines the procedures for carrying out the project's analyses step-by-step.

## References

1. Bailey-Serres, J., Parker, J. E., Ainsworth, E. A., Oldroyd, G., & Schroeder, J. (2019). Genetic strategies for improving crop yields. <em>Nature, 575</em>(7781), 109–118. https://doi.org/10.1038/s41586-019-1679-0
2. Bartholomé, J., Prakash, P. T., & Cobb, J. N. (2022c). Genomic Prediction: Progress and Perspectives for rice improvement. In <em>Methods in molecular biology</em> (pp. 569–617). https://doi.org/10.1007/978-1-0716-2205-6_21
3. Jackson, R. J., Buntjer, J. B., Bentley, A. R., Da Lage, J., Byrne, E., Burt, C., Jack, P., Berry, S., Flatman, E., Poupard, B., Smith, S. P., Hayes, C., Barber, T., Love, B., Gaynor, R. C., Gorjanc, G., Howell, P., Mackay, I., Hickey, J. M., & Ober, E. S. (2023c). Phenomic and genomic prediction of yield on multiple locations in winter wheat. <em>Frontiers in Genetics, 14</em>. https://doi.org/10.3389/fgene.2023.1164935
4. Ontario Ministry of Agriculture, Food and Rural Affairs. (2023, October 23). <em>Cereal Production in Ontario</em>. https://omafra.gov.on.ca/english/crops/field/cereal.html
5. Piepho, H., Laidig, F., Drobek, T., & Meyer, U. (2014). Dissecting genetic and non-genetic sources of long-term yield trend in German official variety trials. <em>Theoretical and Applied Genetics, 127</em>(5), 1009–1018. https://doi.org/10.1007/s00122-014-2275-1
6. So, D., Smith, A., Sparry, E., & Lukens, L. (2022). Genetics, not environment, contributed to winter wheat yield gains in Ontario, Canada. <em>Theoretical and Applied Genetics, 135</em>(6), 1893–1908. https://doi.org/10.1007/s00122-022-04082-3
7. Torkamaneh, D., Laroche, J., Bastien, M., Abed, A., & Belzile, F. (2017). Fast-GBS: a new pipeline for the efficient and highly accurate calling of SNPs from genotyping-by-sequencing data. <em>BMC Bioinformatics, 18</em>(1). https://doi.org/10.1186/s12859-016-1431-9
8. Zhao, H., Lin, Z., Khansefid, M., Tibbits, J., & Hayden, M. J. (2023). Genomic prediction and selection response for grain yield in safflower. <em>Frontiers in Genetics, 14</em>. https://doi.org/10.3389/fgene.2023.1129433