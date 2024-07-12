# ----- LOADING PACKAGES -----

library(cluster)
library(dplyr)
library(factoextra)
library(ggplot2)
library(purrr)
library(stats)
library(VariantAnnotation)
library(vcfR)
library(vegan)

# ----- OBTAINING HWE VALUES -----

# Read the appropriate .vcf.gz files as "CollapsedVCF" objects.
# ZNF23:
znf23_afr <- readVcf('znf23_afr.vcf.gz', verbose = FALSE)
znf23_eas <- readVcf('znf23_eas.vcf.gz', verbose = FALSE)
# FAM135B:
fam135b_afr <- readVcf('fam135b_afr.vcf.gz', verbose = FALSE)
fam135b_eas <- readVcf('fam135b_eas.vcf.gz', verbose = FALSE)
# Rbm38:
rbm38_afr <- readVcf('rbm38_afr.vcf.gz', verbose = FALSE)
rbm38_eas <- readVcf('rbm38_eas.vcf.gz', verbose = FALSE)

# Count the distribution statistics of the SNPs for each gene in each superpopulation.
# ZNF23:
znf23_afr_snps <- snpSummary(znf23_afr)
znf23_eas_snps <- snpSummary(znf23_eas)
# FAM135B:
fam135b_afr_snps <- snpSummary(fam135b_afr)
fam135b_eas_snps <- snpSummary(fam135b_eas)
# Rbm38:
rbm38_afr_snps <- snpSummary(rbm38_afr)
rbm38_eas_snps <- snpSummary(rbm38_eas)

# Combine the data frames containing the distribution statistics of the SNPs for each superpopulation by genes.
# ZNF23:
znf23_snps <- rbind(znf23_afr_snps, znf23_eas_snps)
# FAM135B:
fam135b_snps <- rbind(fam135b_afr_snps, fam135b_eas_snps)
# Rbm38:
rbm38_snps <- rbind(rbm38_afr_snps, rbm38_eas_snps)

# Create a table displaying the HWE values for each gene in each superpopulation.
hwe_table <- data.frame(Gene = c('Zinc Finger Protein 23',
                                 'Family With Sequence Similarity 135 Member B',
                                 'RNA Binding Motif Protein 38'),
                        HWE_Count = c(sum(!is.na(znf23_snps$HWEpvalue < 0.05)),
                                      sum(!is.na(fam135b_snps$HWEpvalue < 0.05)),
                                      sum(!is.na(rbm38_snps$HWEpvalue < 0.05))))
View(hwe_table)

# ----- PROCESSING THE DATA -----

# Read in the appropriate .vcf.gz files as "vcfR" objects.
# ZNF23:
znf23_afr <- read.vcfR('znf23_afr.vcf.gz', convertNA = TRUE)
znf23_eas <- read.vcfR('znf23_eas.vcf.gz', convertNA = TRUE)
# FAM135B:
fam135b_afr <- read.vcfR('fam135b_afr.vcf.gz', convertNA = TRUE)
fam135b_eas <- read.vcfR('fam135b_eas.vcf.gz', convertNA = TRUE)
# Rbm38:
rbm38_afr <- read.vcfR('rbm38_afr.vcf.gz', convertNA = TRUE)
rbm38_eas <- read.vcfR('rbm38_eas.vcf.gz', convertNA = TRUE)

# Extract numerically encoded genotypes and output the results as matrices.
# ZNF23:
znf23_afr_mx <- extract.gt(znf23_afr, element = 'GT', as.numeric = TRUE, IDtoRowNames = TRUE)
znf23_eas_mx <- extract.gt(znf23_eas, element = 'GT', as.numeric = TRUE, IDtoRowNames = TRUE)
# FAM135B:
fam135b_afr_mx <- extract.gt(fam135b_afr, element = 'GT', as.numeric = TRUE, IDtoRowNames = TRUE)
fam135b_eas_mx <- extract.gt(fam135b_eas, element = 'GT', as.numeric = TRUE, IDtoRowNames = TRUE)
# Rbm38:
rbm38_afr_mx <- extract.gt(rbm38_afr, element = 'GT', as.numeric = TRUE, IDtoRowNames = TRUE)
rbm38_eas_mx <- extract.gt(rbm38_eas, element = 'GT', as.numeric = TRUE, IDtoRowNames = TRUE)

# Add the appropriate suffixes to distinguish between individuals belonging to the African and East Asian superpopulations.
# ZNF23:
colnames(znf23_afr_mx) <- paste0(colnames(znf23_afr_mx), '_AFR')
colnames(znf23_eas_mx) <- paste0(colnames(znf23_eas_mx), '_EAS')
# FAM135B:
colnames(fam135b_afr_mx) <- paste0(colnames(fam135b_afr_mx), '_AFR')
colnames(fam135b_eas_mx) <- paste0(colnames(fam135b_eas_mx), '_EAS')
# Rbm38:
colnames(rbm38_afr_mx) <- paste0(colnames(rbm38_afr_mx), '_AFR')
colnames(rbm38_eas_mx) <- paste0(colnames(rbm38_eas_mx), '_EAS')

# For each gene, combine the matrices belonging to the African and East Asian superpopulations, then output the results as data frames.
# ZNF23:
znf23_df <- cbind(znf23_afr_mx, znf23_eas_mx) %>%
  as.data.frame()
# FAM135B:
fam135b_df <- cbind(fam135b_afr_mx, fam135b_eas_mx) %>%
  as.data.frame()
# Rbm38:
rbm38_df <- cbind(rbm38_afr_mx, rbm38_eas_mx) %>%
  as.data.frame()

# For each gene, add a column named "Frequency" containing the allele frequencies per individual.
# ZNF23:
znf23_df$Frequency <- rowSums(znf23_df) / ncol(znf23_df)
# FAM135B:
fam135b_df$Frequency <- rowSums(fam135b_df) / ncol(fam135b_df)
# Rbm38:
rbm38_df$Frequency <- rowSums(rbm38_df) / ncol(rbm38_df)

# For each gene, filter out sites with allele frequencies less than 0.001, drop the column "Frequency," and omit any NAs.
# ZNF23:
znf23_df <- znf23_df %>%
  filter(.$Frequency > 0.001) %>%
  subset(select = -c(.$Frequency)) %>%
  na.omit() # (Bhalla, n.d.).
# FAM135B:
fam135b_df <- fam135b_df %>%
  filter(.$Frequency > 0.001) %>%
  subset(select = -c(.$Frequency)) %>%
  na.omit() # (Bhalla, n.d.).
# Rbm38:
rbm38_df <- rbm38_df %>%
  filter(.$Frequency > 0.001) %>%
  subset(select = -c(.$Frequency)) %>%
  na.omit() # (Bhalla, n.d.).

# Define a function that removes any columns with zero variance in a matrix.
omit_zero_var_columns <- function(mx) {
  sds <- apply(mx, 2, sd, na.rm = TRUE)
  zero_var_columns <- which(sds == 0)
  if (length(zero_var_columns) > 0) {
    mx <- mx[, -zero_var_columns]
  }
  return(mx)
}

# Combine the data frames for all three genes into a single matrix. Then, transpose the matrix such that the chromosome sites make up the column names and the individual IDs make up the row names. Finally, omit any columns in the matrix with zero variance.
all_genes_mx <- rbind(znf23_df, fam135b_df, rbm38_df) %>%
  t() %>%
  omit_zero_var_columns()

# Check for NAs in the matrix.
sum(is.na(all_genes_mx))

# Calculate the means and standard deviations of the data contained in the matrix.
means <- apply(all_genes_mx, 2, mean, na.rm = TRUE) # (finnstats, 2021).
sds <- apply(all_genes_mx, 2, sd, na.rm = TRUE) # (finnstats, 2021).

# Scale the matrix using the means and standard deviations calculated above.
scaled_all_genes_mx <- scale(all_genes_mx, center = means, scale = sds) # (finnstats, 2021).

# Recheck for NAs in the scaled matrix.
sum(is.na(scaled_all_genes_mx))

# Check for the number of observations in the scaled matrix.
nrow(scaled_all_genes_mx)
# The scaled matrix contains 1,052 observations.

# Set a seed to ensure that the analyses downstream produce reproducible results.
set.seed(123)

# ----- FINDING THE OPTIMAL NUMBER OF CLUSTERS -----

# Use the average silhouette method to find out the optimal number of clusters to use in downstream analyses.
fviz_nbclust(x = scaled_all_genes_mx, FUN = hcut, method = 'silhouette') +
  labs(title = 'Fig. 1. The Optimal Number of Clusters Obtained Using the Average Silhouette Method') # (Hierarchical Cluster, n.d.).
# The optimal number of clusters according to the average silhouette method is two.

# ----- GENERATING A SCALED DISSIMILARITY MATRIX -----

# Generate a scaled dissimilarity matrix for use downstream analyses.
scaled_diss_mx <- scale(dist(all_genes_mx, method = 'euclidean'))

# ----- PERFORMING AGGLOMERATIVE HIERARCHICAL CLUSTERING -----

# Define a function that computes the agglomerative coefficients for all six methods available as inputs to the "agnes" function.
agnes_methods <- c('average', 'single', 'complete', 'ward', 'weighted', 'gaverage')
names(agnes_methods) <- c('average', 'single', 'complete', 'ward', 'weighted', 'gaverage')
agnes_coef_calculator <- function(agnes_method) {
  agnes_coef <- agnes(scaled_all_genes_mx, method = agnes_method)$ac
} # (Hierarchical Cluster, n.d.).

# View the agglomerative coefficients for all six methods available as inputs to the "agnes" function.
map_dbl(agnes_methods, agnes_coef_calculator) # (Hierarchical Cluster, n.d.).

# The generalized average method ("gaverage") shows the highest agglomerative coefficient (0.9621833) but is not available as input to the "eclust" function. Therefore, we will use Ward's method ("ward" or "ward.D"), which has the second highest agglomerative coefficient (0.9293915) and is available as input to the "eclust" function.

# Perform agglomerative hierarchical clustering with two clusters.
agnes_clust <- eclust(scaled_diss_mx, FUNcluster = 'agnes', k = 2, hc_method = 'ward.D')

# Visualize the clusters using a scatter plot.
fviz_cluster(agnes_clust, ellipse.type = 'convex', palette = 'jco', geom = 'point', ggtheme = theme_minimal()) +
  ggtitle('Fig. 2. A Scatter Plot Showing the Clusters Obtained Using Agglomerative Hierarchical Clustering') # (Hierarchical Cluster, n.d.).

# Validate the clusters using a silhouette plot.
fviz_silhouette(agnes_clust, print.summary = TRUE, palette = 'jco', ggtheme = theme_minimal())

# ----- PERFORMING DIVISIVE HIERARCHICAL CLUSTERING -----

# Perform divisive hierarchical clustering with two clusters.
diana_clust <- eclust(scaled_diss_mx, FUNcluster = 'diana', k = 2)

# Visualize the clusters using a scatter plot.
fviz_cluster(diana_clust, ellipse.type = 'convex', palette = 'jco', geom = 'point', ggtheme = theme_minimal()) +
  ggtitle('Fig. 3. A Scatter Plot Showing the Clusters Obtained Using Divisive Hierarchical Clustering') # (Hierarchical Cluster, n.d.).

# Validate the clusters using a silhouette plot.
fviz_silhouette(diana_clust, palette = 'jco', ggtheme = theme_minimal())

# ----- PERFORMING K-MEANS CLUSTERING -----

# Perform K-means clustering with two clusters.
k_means_clust <- eclust(scaled_diss_mx, FUNcluster = 'kmeans', k = 2)

# Visualize the clusters using a scatter plot.
fviz_cluster(k_means_clust, ellipse.type = 'convex', palette = 'jco', geom = 'point', ggtheme = theme_minimal()) +
  ggtitle('Fig. 4. A Scatter Plot Showing the Clusters Obtained Using K-Means Clustering') # (Alboukadel, 2019).

# Validate the clusters using a silhouette plot.
fviz_silhouette(k_means_clust, palette = 'jco', ggtheme = theme_minimal())

# ----- PERFORMING PAM CLUSTERING -----

# Perform PAM clustering with two clusters.
pam_clust <- eclust(scaled_diss_mx, FUNcluster = 'pam', k = 2)

# Visualize the clusters using a scatter plot.
fviz_cluster(pam_clust, ellipse.type = 'convex', palette = 'jco', geom = 'point', ggtheme = theme_minimal()) +
  ggtitle('Fig. 5. A Scatter Plot Showing the Clusters Obtained Using PAM Clustering') # (Alboukadel, 2019).

# Validate the clusters using a silhouette plot.
fviz_silhouette(pam_clust, palette = 'jco', ggtheme = theme_minimal())

# ----- REFERENCES -----

# Alboukadel. (2019, December 25). Types of Clustering Methods: Overview and Quick Start R Code. Datanovia. https://www.datanovia.com/en/blog/types-of-clustering-methods-overview-and-quick-start-r-code/
# Bhalla, D. (n.d.). R : KEEP / DROP COLUMNS FROM DATA FRAME. Listen Data. https://www.listendata.com/2015/06/r-keep-drop-columns-from-data-frame.html
# finnstats. (2021, April 20). Cluster Analysis in R. R-bloggers. https://www.r-bloggers.com/2021/04/cluster-analysis-in-r/
# Hierarchical Cluster Analysis. (n.d.). UC Business Analytics R Programming Guide. Retrieved April 5, 2023, from https://uc-r.github.io/hc_clustering
