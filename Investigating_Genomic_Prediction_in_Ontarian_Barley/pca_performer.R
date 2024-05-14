# Title                   : Performing Principal Component Analysis on Ontarian Barley Varieties
# Author                  : Abelhard Jauwena
# Student Number          : 1040413
# Course Code             : BINF*6999
# Course Name             : Bioinformatics Master's Project
# Submission Date         : December 15, 2023
# University Advisors     :
  # Informatics Expertise : Dr. Yan Yan, School of Computer Science.
  # Biological Expertise  : Dr. Lewis Lukens, Department of Plant Agriculture

# ----- Instructions -----

# This script:
  # 1) Explores the dataset containing called, filtered, and imputed SNPs from Ontarian barley varieties.
  # 2) Elucidates the population genetic structure of these barley varieties using principal component analysis (PCA).

# ----- Preparations -----

# Set the working directory to the one that has the dataset containing the called SNPs.

# Load the appropriate packages.
library(dplyr)
library(emmeans) # Contains the "emmeans()" function for calculating the adjusted year and entry means.
library(naniar) # Contains the "miss_var_summary()" function.
library(patchwork)
library(RCurl)
library(readxl) # For reading in the metadata file.
library(SNPRelate)
library(stringr)
library(tidyverse)
library(vcfR)

# Set the seed.
set.seed(1000)

# Set the theme.
theme_set(theme_minimal(base_size = 14) +
            theme(legend.direction = "vertical",
                  legend.position = "top",
                  legend.key.width = unit(2, "cm")))

# ----- Obtaining the Varieties' Generations in the Metadata -----

# Read in the metadata file as a data frame.
df_metadata <- read_excel("data_AllOCCCBarley_copy.xlsx") %>%
  as_tibble()
# The metadata file is the dataset containing the unfiltered historical barley yields. Low-count barley varieties have not yet been removed.

# Check for any missing entries in each column.
miss_var_summary(df_metadata)
# 87 entries were missing from the column "Yield," which corresponds to only ~0.559% of the total entries.

# Obtain the metadata's column names.
names(df_metadata)

# Create a data frame containing the years of first entry (YFE) of each variety, where applicable.
df_metadata_varieties_and_yfe <- df_metadata %>%
  # Retain only rows corresponding to unique year-variety combinations ("distinct()" is case-sensitive and removes the rest of the columns).
  distinct(Year, Variety) %>%
  # Arrange the rows by year (from oldest to most recent) then by variety in alphabetical order.
  arrange(Year, Variety) %>%
  # Group the tibble by variety for downstream operations (there are no observable effects).
  group_by(Variety) %>%
  # For each variety, only retain the first row, which corresponds to its YFE.
  filter(row_number() == 1) %>%
  # Remove the variety grouping from the tibble (there are no observable effects).
  ungroup() %>%
  # Add a column containing the YFE for each variety as integers.
  mutate(Year_of_First_Entry = as.integer(Year)) %>%
  # Remove the original column containing years.
  select(-Year)

# View all the unique years of first entry in the metadata.
unique(df_metadata_varieties_and_yfe$Year_of_First_Entry)
# The years of first entry range from 1958 to 2021. As such, the cutoff year is set as 1990 (midway between 1958 and 2021). This cutoff year distinguishes between old and new varieties.

# Create a data frame for the metadata that separates old and new varieties.
df_metadata_varieties_and_generations <- df_metadata_varieties_and_yfe %>%
  # Create a column, "Generation," distinguishing varieties whose year is first entry is before and after 1990.
  mutate(Generation = case_when(Year_of_First_Entry >= 1990 ~ "Post-1990",
                                Year_of_First_Entry < 1990 ~ "Pre-1990")) %>% # (So, n.d.).
  # Retain only rows corresponding to unique variety-generation combinations ("distinct()" is case-sensitive and removes the "Year_of_First_Entry" column).
  distinct(Variety, Generation)

# ----- Checking for Overlapping Varieties in the SNP and Yield Datasets -----

# Read in the SNP dataset as a Genomic Data Structure (GDS) object, then output the file into the current working directory.
snpgdsVCF2GDS("barley_merged_platypus_loci_imputed.vcf", "barley_merged_platypus_loci_imputed_converted.gds", method = "biallelic.only") # (So, n.d.).

# Open the GDS object
gds_snps <- snpgdsOpen("barley_merged_platypus_loci_imputed_converted.gds") # (So, n.d.).

# Extract a genotype matrix from the GDS object.
mat_snps <- snpgdsGetGeno(gds_snps, with.id = TRUE) # (So, n.d.).
# There are 116 varieties x 35,481 SNPs.

# Close the GDS object
closefn.gds(gds_snps) # (So, n.d.).

# Check varieties that overlap between the genotype matrix and metadata.
print(intersect(mat_snps$sample.id, unique(df_metadata$Variety))) %>%
  length()
# 54 out of 116 varieties overlap.

# Check varieties that are present in the genotype matrix but not in the metadata.
print(setdiff(mat_snps$sample.id, unique(df_metadata$Variety))) %>%
  length()
# Correspondingly, 62 out of 116 varieties do not overlap.

# Check whether some varieties are present in the metadata (these varieties are further divided into sub-varieties in the genotype matrix).
sum(str_detect(df_metadata$Variety, "Conestogo")) > 0 # True.
sum(str_detect(df_metadata$Variety, "Metcalfe")) > 0 # False.
sum(str_detect(df_metadata$Variety, "09N2-31")) > 0 # False.

# Join the variety IDs from the genotype matrix with their respective generations, then output the results as a data frame.
df_snp_data_varieties_and_generations <- mat_snps$sample.id %>%
  # Output the variety IDs as a data frame.
  data.frame("Variety" = .) %>%
  # Add the column corresponding to the varieties' generations.
  left_join(df_metadata_varieties_and_generations, by = c("Variety")) # (So, n.d.).

# Create an empty vector to store varieties with a known (non-NA) generation.
vec_overlapping_varieties_and_generations <- c()

# Loop through each row in the data frame that contains varieties and their generations.
for (i in 1:nrow(df_snp_data_varieties_and_generations)) {
  # If the "Generation" column does not contain an "NA"...
  if (!is.na(df_snp_data_varieties_and_generations[i, 2])) {
    # Append the sample in the "Variety" column to the empty vector.
    vec_overlapping_varieties_and_generations <- append(vec_overlapping_varieties_and_generations, df_snp_data_varieties_and_generations[i, 1])
  }
}

# Check the number of varieties with a known generation.
length(vec_overlapping_varieties_and_generations)
# 54 out of 116 varieties have a known generation.

# Double-check the number of varieties with a known generation from the original data frame.
df_snp_data_varieties_and_generations$Generation[!is.na(df_snp_data_varieties_and_generations$Generation)] %>%
  length()
# Also 54.

# Save the varieties with a known generation as a .txt file for use in the script "b_06_sample_subsetter_merged.sh" in Compute Canada.
write.table(vec_overlapping_varieties_and_generations, file = "overlapping_varieties.txt", quote = FALSE, row.names = FALSE, col.names = FALSE)

# ----- Subsetting Overlapping Varieties -----

# Varieties were subsetted in Compute Canada using the scripts "b_05_tabix_indexer_merged.sh" (using Tabix) and "b_06_sample_subsetter_merged.sh" (using BCFtools). The results were outputted into a file called "barley_merged_platypus_loci_imputed_subsetted.vcf."

# ----- Obtaining the Proportion of Variation Explained by Each PC -----

# Read in the subsetted SNP dataset as a GDS object, then output the file into the current working directory.
snpgdsVCF2GDS("barley_merged_platypus_loci_imputed_subsetted.vcf", "barley_merged_platypus_loci_imputed_subsetted_converted.gds", method = "biallelic.only") # (So, n.d.).

# Check the total number of varieties and SNPs in the GDS object.
snpgdsSummary("barley_merged_platypus_loci_imputed_subsetted_converted.gds") # (So, n.d.).
# There are 54 varieties and 35,481 SNPs.

# Open the GDS object
gds_snps_subset <- snpgdsOpen("barley_merged_platypus_loci_imputed_subsetted_converted.gds", readonly = FALSE) # (So, n.d.).

# Extract a genotype matrix from the GDS object.
mat_snps_subset <- snpgdsGetGeno(gds_snps_subset, with.id = TRUE) # (So, n.d.).
# There are 54 varieties x 35,481 SNPs.

# Create another matrix containing only the SNPs of all the varieties from the genotype matrix.
mat_snps_subset_geno <- mat_snps_subset$genotype # (So, n.d.).

# Perform PCA on this matrix, returning the results as a "prcomp" object.
pca_a <- prcomp(mat_snps_subset_geno, scale = FALSE, center = TRUE) # (So, n.d.).

# View the eigenvalues from the prcomp object to see how much variation each PC explains.
tibble(pca_a$sdev^2) # (So, n.d.).

# Calculate the proportion of variation for each PC by dividing the ith PC's variation by the total allelic variation.
tibble(proportion_variation = pca_a$sdev^2 / sum(pca_a$sdev^2) * 100) # (So, n.d.).
# PC1 explains most (26.2%) of the total allelic variation, while PC2 explains 9.33%. PC1 will thus be plotted against PC2 to elucidate the population genetic structure of the barley varieties.

# ----- Performing PCA -----

# Obtain the sample IDs in the GDS object.
vec_sample_ids <- read.gdsn(index.gdsn(gds_snps_subset, "sample.id")) # (Lama, 2018).

# Prune SNPs based on LD. Try using a linkage disequilibrium (LD) threshold of 0.1 (Lama, 2018).
list_snp_set <- snpgdsLDpruning(gds_snps_subset, autosome.only = FALSE, remove.monosnp = TRUE, ld.threshold = 0.1, verbose = FALSE) # (Lama, 2018).

# Store the selected SNP IDs in an object.
int_snp_set_ids <- unlist(list_snp_set) # (Smith, 2023).
length(int_snp_set_ids)
# 7,813 SNPs remained after pruning with an LD threshold of 0.1 and the set seed.

# Perform PCA again on the LD-pruned SNPs.
pca_b <- snpgdsPCA(gds_snps_subset, sample.id = NULL, snp.id = int_snp_set_ids, autosome.only = FALSE, verbose = TRUE) # (Lama, 2018; Smith, 2023).

# Create a data frame containing the principal components (PCs) of each variety (with eigenvectors), then rename the columns.
df_snp_data_varieties_and_pcs <- data.frame(entry = pca_b$sample.id, pca_b$eigenvect) # (Smith, 2023).
colnames(df_snp_data_varieties_and_pcs)[1] <- "Variety" # (Smith, 2023).
colnames(df_snp_data_varieties_and_pcs)[c(2:33)] <- gsub("X", "PC", colnames(df_snp_data_varieties_and_pcs)[c(2:33)]) # (Smith, 2023).

# Check the metadata's column names again.
names(df_metadata)

# Create a data frame containing the years and yields of each variety, where applicable.
df_metadata_varieties_years_and_yields <- df_metadata %>%
  select(c(Variety, Year, Yield)) %>%
  filter(Variety %in% pca_b$sample.id)

# Group yields by variety names.
df_metadata_varieties_years_and_yields <- df_metadata_varieties_years_and_yields %>%
  # Arrange the rows by variety (in alphabetical order), then by year (from oldest to most recent).
  arrange(Variety, Year) %>%
  # Group the tibble by variety for downstream operations (there are no observable effects).
  group_by(Variety)

# Convert the varieties into factors.
df_metadata_varieties_years_and_yields$Variety <- as.factor(df_metadata_varieties_years_and_yields$Variety)

# Linearly regress yields by varieties, where varieties are treated as factor levels.
fit_lm_varieties_and_yields <- lm(Yield ~ Variety, df_metadata_varieties_years_and_yields)

# Calculate the adjusted yield means per variety and output the results as a data frame.
df_group_adjusted_means_varieties_and_yields <- emmeans(fit_lm_varieties_and_yields, "Variety", rg.limit = 15000) %>%
  broom::tidy()

# Change the column named "estimate" in the data frame to "Adjusted Yield Mean."
colnames(df_group_adjusted_means_varieties_and_yields)[which(names(df_group_adjusted_means_varieties_and_yields) == "estimate")] <- "Adjusted_Yield_Mean"

# Copy the information about barley types (i.e., six-row vs. two-row barley) from the Government of Canada website (Government of Canada, 2022). Then, read it into R as a tab-delimited tibble and only retain information corresponding to variety names.
df_cfia_varieties_and_types <- read_delim("barley_types_manual.txt", delim = ",", col_names = c("Variety", "Type"))

# Merge the data frames containing each variety's 1) adjusted yield mean, 2) PCs, 3) generation, and 4) type into one data frame.
df_ggplot_varieties_yields_and_pcs <- inner_join(df_group_adjusted_means_varieties_and_yields, df_snp_data_varieties_and_pcs) %>%
  left_join(df_snp_data_varieties_and_generations) %>%
  left_join(df_cfia_varieties_and_types)
# This data frame will be used for plotting.

# Also, create a data frame containing the year of first entry (YFE) and yields of each variety for downstream use.
df_metadata_varieties_and_yfe_2 <- df_metadata_varieties_years_and_yields %>%
  # For each variety, only retain the first row, which corresponds to its YFE (the data frame is already grouped by variety).
  filter(row_number() == 1) %>%
  # Add a column containing the YFE for each variety as integers.
  mutate(Year_of_First_Entry = as.integer(Year)) %>%
  # Remove the original column containing years.
  select(-Year) %>%
  # Remove the original column containing yields.
  select(-Yield)

# ----- Plotting PCA Results -----

# Plot PC1 against PC2, differentiating points based on both generation and row type.
ggplot(data = df_ggplot_varieties_yields_and_pcs, aes(x = PC1, y = PC2, colour = Generation, shape = Type)) +
  geom_point(size = 3) +
  labs(colour = "Generation", shape = "Type") +
  theme_light() # (Smith, 2023).
# PC1 captures the different barley row types (i.e., six-row versus two-row). On the other hand, PC2 likely captures the year in which a variety is introduced.

# Plot PC1 against adjusted yield means, differentiating points based on both generation and row type.
ggplot(data = df_ggplot_varieties_yields_and_pcs, aes(x = PC1, y = Adjusted_Yield_Mean, colour = Generation, shape = Type)) +
  geom_point(size = 3) +
  labs(colour = "Generation", shape = "Type") +
  theme_light() +
  ylab("Adjusted Yield Mean (kg/ha)") # (Smith, 2023).
# New (post-1990) varieties seem to confer greater yields than old (pre-1990) varieties, suggesting genetic improvement. On the other hand, both six-row and two-row barley varieties seem to not show significant differences in yields.

# Close the GDS object
closefn.gds(gds_snps_subset)

# ----- Pruning (Again) and Converting the SNP Dataset -----

# The SNPs need to be pruned again and converted to the "012" format using VCFtools (Smith, 2023). Doing so in VCFtools requires a text file with a list of IDs, chromosomes, and positions in the subsetted SNPs.

# In the 012 format, "0" represents a homozygous reference genotype (with no copies of the non-reference allele), "1" represents a heterozygous genotype (with one copy of the non-reference allele), and "2" represents a homozygous alternate genotype (with two copies of the non-reference allele) (Auton & Marcketta, 2015; Strandén & Christensen, 2011).

# Read in the VCF file as a "vcfR" object.
vcfR_vcf <- read.vcfR("barley_merged_platypus_loci_imputed_subsetted.vcf") # (Smith, 2023).
# 35,876 SNPs were identified.

# Store the fixed data from the vcfR object as a data frame.
df_vcf_fix <- as.data.frame(vcfR_vcf@fix) # (Smith, 2023).

# Subset the data frame to only include variants whose SNPs survived pruning.
df_vcf_fix_prune <- df_vcf_fix %>%
  filter(row.names(df_vcf_fix) %in% int_snp_set_ids) # (Smith, 2023).

# Create a data frame containing the IDs of the SNPs that survived pruning, then save the data frame as a tab-delimited file.
df_snp_data_ids_prune <- df_vcf_fix_prune %>%
  select(ID) # (Smith, 2023).
write_delim(df_snp_data_ids_prune, "pruned_snp_ids.txt", delim = "\t", col_names = FALSE) # (Smith, 2023).

# Create a data frame containing the chromosomes and positions of the SNPs that survived pruning, then save the data frame as a tab-delimited file.
df_snp_data_chromosomes_and_positions_prune <- df_vcf_fix_prune %>%
  select(CHROM, POS) # (Smith, 2023).
write_delim(df_snp_data_chromosomes_and_positions_prune, "pruned_snp_chromosomes_and_positions.txt", delim = "\t", col_names = FALSE) # (Smith, 2023).

# SNPs were pruned and converted to the 012 format in Compute Canada using the script "b_07_snp_pruner_and_converter_merged.sh" (Smith, 2023).

# ----- Preparing Data for Genomic Prediction -----

# Read in the calls, variety, and position dataset as data frames.
df_snp_data_012_calls <- read.delim("barley_merged_platypus_loci_imputed_pruned_subsetted_012.012", header = FALSE) # (Smith, 2023).
df_snp_data_012_varieties <- read.delim("barley_merged_platypus_loci_imputed_pruned_subsetted_012.012.indv", header = FALSE) # (Smith, 2023).
df_snp_data_012_positions <- read.delim("barley_merged_platypus_loci_imputed_pruned_subsetted_012.012.pos", header = FALSE) # (Smith, 2023).

# Create a genotype table, filling in the first column with variety names.
df_snp_data_012_geno_table <- df_snp_data_012_calls %>%
  mutate(V1 = df_snp_data_012_varieties$V1) # (Smith, 2023).
# Then, rename the first column as "Variety" and the rest of the columns as chromosome positions.
colnames(df_snp_data_012_geno_table) <- c("Variety", paste0(df_snp_data_012_positions$V1, ":", df_snp_data_012_positions$V2)) # (Smith, 2023).

# Save the genotype table as a .csv file.
write.csv(df_snp_data_012_geno_table, "barley_merged_platypus_loci_imputed_subsetted_012_geno_table.csv", row.names = FALSE) # (Smith, 2023).

# Create a data frame that will be a combination of i) the genotype table, ii) the data frame containing the adjusted yield means per variety, and iii) data frame containing the year of first entry (YFE) and yields of each variety. First, remove the column containing varieties from the data frame.
df_snp_data_012_converted_with_adjusted_yield_means_and_yfe <- df_snp_data_012_geno_table[, -c(1)] # (Smith, 2023).

# Then, subtract the calls in the data frame by one to convert it to the "-101" format.
df_snp_data_012_converted_with_adjusted_yield_means_and_yfe <- df_snp_data_012_converted_with_adjusted_yield_means_and_yfe - 1 # (Smith, 2023).
# In the -101 format, "-1" represents a homozygous reference genotype (with no copies of the non-reference allele), "0" represents a heterozygous genotype (with one copy of the non-reference allele), and "1" represents a homozygous alternate genotype (with two copies of the non-reference allele) (Auton & Marcketta, 2015; Strandén & Christensen, 2011).

# Then, add the column containing varieties back to the data frame for grouping.
df_snp_data_012_converted_with_adjusted_yield_means_and_yfe <- df_snp_data_012_converted_with_adjusted_yield_means_and_yfe %>%
  mutate(Variety = df_snp_data_012_geno_table$Variety) %>%
  # Move the column containing varieties to the leftmost side.
  relocate(Variety)

# Finally, join the data frame with i) the data frame containing the adjusted yield means per variety and ii) data frame containing the year of first entry (YFE) and yields of each variety.
df_snp_data_012_converted_with_adjusted_yield_means_and_yfe <- left_join(df_snp_data_012_converted_with_adjusted_yield_means_and_yfe, df_group_adjusted_means_varieties_and_yields) %>%
  left_join(df_metadata_varieties_and_yfe_2) %>%
  # Remove the column containing varieties.
  select(-Variety)

# Change the row names of the data frame into variety names.
rownames(df_snp_data_012_converted_with_adjusted_yield_means_and_yfe) <- df_snp_data_012_geno_table$Variety

# Save the data frame as a .csv file.
write.csv(df_snp_data_012_converted_with_adjusted_yield_means_and_yfe, "barley_merged_platypus_loci_imputed_subsetted_-101_mat_with_adjusted_yield_means_and_yfe.csv") # (Smith, 2023).

# ----- References -----

# Auton, A., & Marcketta, A. (2015). VCFtools. https://vcftools.github.io/man_0112a.html
# Government of Canada. (2022, June 7). Varieties of Crop Kinds Registered in Canada. https://active.inspection.gc.ca/netapp/regvar/regvar_lookupe.aspx
# kassambara. (2017, October 8). Principal Component Analysis in R: prcomp vs princomp. STHDA. http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/
# Lama, T. (2018, May 18). Revised Red Squirrel: Data prep, LD-based SNP pruning, PCA, IBD. Revised Red Squirrel: Data prep, LD-based SNP pruning, PCA, IBD. https://rstudio-pubs-static.s3.amazonaws.com/407692_672504588334488fa63007155c382f70.html
# Principal Components Analysis. (n.d.). ETH Zurich. Retrieved October 5, 2023, from https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html
# Smith, A. (2023, June 21). Prune SNPs and do PCA (again... hopefully last time) [R script]. The R Foundation.
# So, D. (n.d.). Chapter 4 - SNP Characteristics and Population Structure of OCCC Winter Wheat Cultivars [R script]. The R Foundation.
# Strandén, I., & Christensen, O. (2011). Allele coding in genomic evaluation. Genetics Selection Evolution, 43(1). https://doi.org/10.1186/1297-9686-43-25
