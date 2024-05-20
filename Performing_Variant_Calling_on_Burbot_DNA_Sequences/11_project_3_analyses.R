# --- Load the appropriate package(s) ---
library(dplyr)
library(tidyverse)

# --- NUMBER OF SNPS ---

# Read in the file corresponding to the number of SNPs identified using both software as a data frame.
snp_number_df <- read.table('snp_number.txt',
                            sep = '\t',
                            header = FALSE)

# Set the values in the first row as column names.
names(snp_number_df) <- snp_number_df[1, ]
snp_number_df <- snp_number_df[-1, ]

# View the number of SNPs identified using both software.
View(snp_number_df)

# --- SNP OVERLAP ---

# Read in the file corresponding to the overlap of SNPs as a data frame.
snp_overlap_df <- read.table('snp_overlap.txt',
                             sep = '\t',
                             header = FALSE)

# Set the values in the first row as column names.
names(snp_overlap_df) <- snp_overlap_df[1, ]
snp_overlap_df <- snp_overlap_df[-1, ]

# View the overlap of SNPs.
View(snp_overlap_df)

# --- MINOR ALLELE FREQUENCY: BCFTOOLS MPILEUP ---

# Read in the file corresponding to the allele frequencies of all SNPs identified using BCFtools mpileup.
allele_freq_df_bcftools <- read.table('results_bcftools_filtered_minor_allele_frequency.frq',
                                      sep = '\t',
                                      header = FALSE,
                                      row.names = NULL,
                                      fill = TRUE)
allele_freq_df_bcftools <- allele_freq_df_bcftools[-1, ]
colnames(allele_freq_df_bcftools) <- c('CHROM', 'POS', 'N_ALLELES', 'N_CHR', 'FREQ_1', 'FREQ_2')
# Note: Reading .frq files into R is rather difficult because there is no header for the sixth column. As such, I proceeded to set "header" as "FALSE," "row.names" as "NULL," and "fill" as "TRUE." I then removed the first row and wrote the column names manually. I also used this approach for the .frq file obtained from using Freebayes.

# Convert the allele frequencies into numeric values.
allele_freq_df_bcftools$FREQ_1 <- as.numeric(allele_freq_df_bcftools$FREQ_1)
allele_freq_df_bcftools$FREQ_2 <- as.numeric(allele_freq_df_bcftools$FREQ_2)

# Subset the columns corresponding to the allele frequencies identified using BCFtools mpileup into a separate data frame. I will loop through this data frame later.
allele_freq_subset_df_bcftools <- allele_freq_df_bcftools %>%
  select(FREQ_1, FREQ_2)

# Create an empty vector that will contain all the allele frequencies identified using BCFtools mpileup (i.e., both major and minor).
maf_vec_bcftools <- c()

# Loop through each row in the data frame subset.
for (i in 1:nrow(allele_freq_subset_df_bcftools)) {
  # Loop through each column in each row in the data frame subset.
  for (j in 1:ncol(allele_freq_subset_df_bcftools)) {
    # Append the allele frequency to the vector created above.
    maf_vec_bcftools <- append(x = maf_vec_bcftools,
                               values = allele_freq_subset_df_bcftools[i, j])
  }
}

# Create another data frame containing all the allele frequencies identified using BCFtools mpileup.
maf_df_bcftools <- data.frame(maf_vec_bcftools)

# Plot a histogram showing the distribution of minor allele frequencies across all SNPs identified using BCFtools mpileup.
maf_df_bcftools %>%
  drop_na(maf_vec_bcftools) %>%
  ggplot(aes(x = maf_vec_bcftools)) +
  geom_histogram(color = 4,
                 fill = 'black',
                 binwidth = 0.05) +
  ggtitle('Fig. 1. A histogram of the distribution of minor allele frequencies across all SNPs identified using BCFtools mpileup') +
  xlab('Minor Allele Frequency') +
  ylab('SNP Count') +
  xlim(0, 0.5) +
  ylim(0, 10000)

# --- MINOR ALLELE FREQUENCY: FREEBAYES ---

# Read in the file corresponding to the allele frequencies across all SNPs identified using Freebayes.
allele_freq_df_freebayes <- read.table('results_freebayes_filtered_minor_allele_frequency.frq',
                                       sep = '\t',
                                       header = FALSE,
                                       row.names = NULL,
                                       fill = TRUE)
allele_freq_df_freebayes <- allele_freq_df_freebayes[-1, ]
colnames(allele_freq_df_freebayes) <- c('CHROM', 'POS', 'N_ALLELES', 'N_CHR', 'FREQ_1', 'FREQ_2')

# Convert the allele frequencies into numeric values.
allele_freq_df_freebayes$FREQ_1 <- as.numeric(allele_freq_df_freebayes$FREQ_1)
allele_freq_df_freebayes$FREQ_2 <- as.numeric(allele_freq_df_freebayes$FREQ_2)

# Subset the columns corresponding to the allele frequencies identified using Freebayes into a separate data frame. I will loop through this data frame later.
allele_freq_subset_df_freebayes <- allele_freq_df_freebayes %>%
  select(FREQ_1, FREQ_2)

# Create an empty vector that will contain all the allele frequencies identified using Freebayes (i.e., both major and minor).
maf_vec_freebayes <- c()

# Loop through each row in the data frame subset.
for (i in 1:nrow(allele_freq_subset_df_freebayes)) {
  # Loop through each column in each row in the data frame subset.
  for (j in 1:ncol(allele_freq_subset_df_freebayes)) {
    # Append the allele frequency to the vector created above.
    maf_vec_freebayes <- append(x = maf_vec_freebayes,
                                 values = allele_freq_subset_df_freebayes[i, j])
  }
}

# Create another data frame containing all the allele frequencies identified using Freebayes.
maf_df_freebayes <- data.frame(maf_vec_freebayes)

# Plot a histogram showing the distribution of minor allele frequencies across all SNPs identified using BCFtools mpileup.
maf_df_freebayes %>%
  drop_na(maf_vec_freebayes) %>%
  ggplot(aes(x = maf_vec_freebayes)) +
  geom_histogram(color = 2,
                 fill = 'black',
                 binwidth = 0.05) +
  ggtitle('Fig. 2. A histogram of the distribution of minor allele frequencies across all SNPs identified using Freebayes') +
  xlab('Minor Allele Frequency') +
  ylab('SNP Count') +
  xlim(0, 0.5) +
  ylim(0, 10000)

# --- SUMMED DEPTH: BCFTOOLS MPILEUP ---

# Read in the file corresponding to the depths of all SNPs (summed across all 10 individuals) identified using BCFtools mpileup as a data frame.
summed_depth_df_bcftools <- read.table('results_bcftools_filtered_summed_depth.ldepth',
                                       sep = '\t',
                                       header = FALSE)

# Set the values in the first row as column names.
names(summed_depth_df_bcftools) <- summed_depth_df_bcftools[1, ]
summed_depth_df_bcftools <- summed_depth_df_bcftools[-1, ]

# Convert the summed depths into numeric values.
summed_depth_df_bcftools$SUM_DEPTH <- as.numeric(summed_depth_df_bcftools$SUM_DEPTH)

# Plot a histogram showing the distribution of summed depths across all SNPs identified using BCFtools mpileup.
summed_depth_df_bcftools %>%
  drop_na(SUM_DEPTH) %>%
  ggplot(aes(x = SUM_DEPTH)) +
  geom_histogram(color = 4,
                 fill = 'black') +
  ggtitle('Fig. 3. A histogram of the distribution of depths across all SNPs (summed across all 10 individuals) identified using BCFtools mpileup') +
  xlab('Depth') +
  ylab('SNP Count') +
  xlim(0, 6000) +
  ylim(0, 25000)

# --- SUMMED DEPTH: FREEBAYES ---

# Read in the file corresponding to the depths across all SNPs (summed across all 10 individuals) identified using Freebayes as a data frame.
summed_depth_df_freebayes <- read.table('results_freebayes_filtered_summed_depth.ldepth',
                                        sep = '\t',
                                        header = FALSE)

# Set the values in the first row as column names.
names(summed_depth_df_freebayes) <- summed_depth_df_freebayes[1, ]
summed_depth_df_freebayes <- summed_depth_df_freebayes[-1, ]

# Convert the summed depths into numeric values.
summed_depth_df_freebayes$SUM_DEPTH <- as.numeric(summed_depth_df_freebayes$SUM_DEPTH)

# Plot a histogram showing the distribution of summed depths across all SNPs identified using Freebayes.
summed_depth_df_freebayes %>%
  drop_na(SUM_DEPTH) %>%
  ggplot(aes(x = SUM_DEPTH)) +
  geom_histogram(color = 2,
                 fill = 'black') +
  ggtitle('Fig. 4. A histogram of the distribution of depths across all SNPs (summed across all 10 individuals) identified using Freebayes') +
  xlab('Depth') +
  ylab('SNP Count') +
  xlim(0, 6000) +
  ylim(0, 25000)

# --- MEAN DEPTH: BCFTOOLS MPILEUP ---

# Read in the file corresponding to the mean depths of all SNPs (measured in reads per locus per individual) identified using BCFtools mpileup.
mean_depth_df_bcftools <- read.table('results_bcftools_filtered_mean_depth.ldepth.mean',
                                     sep = '\t',
                                     header = FALSE)

# Set the values in the first row as column names.
names(mean_depth_df_bcftools) <- mean_depth_df_bcftools[1, ]
mean_depth_df_bcftools <- mean_depth_df_bcftools[-1, ]

# Convert the mean depths into numeric values.
mean_depth_df_bcftools$MEAN_DEPTH <- as.numeric(mean_depth_df_bcftools$MEAN_DEPTH)

# Plot a histogram showing the distribution of mean depths across all SNPs identified using BCFtools mpileup.
mean_depth_df_bcftools %>%
  drop_na(MEAN_DEPTH) %>%
  ggplot(aes(x = MEAN_DEPTH)) +
  geom_histogram(color = 4,
                 fill = 'black') +
  ggtitle('Fig. 5. A histogram of the distribution of mean depths across all SNPs (measured in reads per locus per individual) identified using BCFtools mpileup') +
  xlab('Mean Depth') +
  ylab('SNP Count') +
  xlim(0, 600) +
  ylim(0, 25000)

# --- MEAN DEPTH: FREEBAYES ---

# Read in the file corresponding to the mean depths of all SNPs (measured in reads per locus per individual) identified using Freebayes.
mean_depth_df_freebayes <- read.table('results_freebayes_filtered_mean_depth.ldepth.mean',
                                      sep = '\t',
                                      header = FALSE)

# Set the values in the first row as column names.
names(mean_depth_df_freebayes) <- mean_depth_df_freebayes[1, ]
mean_depth_df_freebayes <- mean_depth_df_freebayes[-1, ]

# Convert the mean depths into numeric values.
mean_depth_df_freebayes$MEAN_DEPTH <- as.numeric(mean_depth_df_freebayes$MEAN_DEPTH)

# Plot a histogram showing the distribution of mean depths across all SNPs identified using BCFtools mpileup.
mean_depth_df_freebayes %>%
  drop_na(MEAN_DEPTH) %>%
  ggplot(aes(x = MEAN_DEPTH)) +
  geom_histogram(color = 2,
                 fill = 'black') +
  ggtitle('Fig. 6. A histogram of the distribution of mean depths across all SNPs (measured in reads per locus per individual) identified using Freebayes') +
  xlab('Mean Depth') +
  ylab('SNP Count') +
  xlim(0, 600) +
  ylim(0, 25000)
