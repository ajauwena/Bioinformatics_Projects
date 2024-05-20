# BINF 6110 - Project 4
# NAME: Abelhard Jauwena
# STUDENT ID: 1040413
# TOPIC: Performing Discriminant Analysis of Principal Components

# --- PREPARING R ---

# Load the appropriate packages.
library(adegenet) # (Jombart, 2008; Jombart & Ahmed, 2011).
library(dplyr) # (Wickham et al., 2023).
library(janitor) # (Firke, 2023).
library(openxlsx) # (Schauberger & Walker, 2023).
library(tidyverse) # (Wickham et al., 2019).

# --- PROCESSING DATA ---

# Read the data set as a data frame.
df_data <- read.xlsx(xlsxFile = 'ethiopian_barley_data_set.xlsx', fillMergedCells = TRUE, colNames = TRUE)

# Set the data in the first row as column names and delete the last two columns (i.e., the columns "Loci" and "Code"), which contain loci annotations.
df_data <- row_to_names(df_data, row_number = 1) %>% # (zek19, 2019)
  subset(., select = -c(Loci, Code))

# Subset the data frame such that it only contains data for account numbers, regions of origin, altitude classes, collection years, and SSR markers (Dido et al., 2021).
df_data <- df_data[, c('Acc.No', 'Region', 'Altitude', 'Year collected', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M26', 'M27', 'M28', 'M29', 'M30', 'M31', 'M32', 'M33', 'M34', 'M35', 'M36', 'M37', 'M38', 'M39', 'M40', 'M41', 'M42', 'M43', 'M44', 'M45', 'M46', 'M47', 'M48', 'M49')]

# Convert the data in the column "Year collected" into factors to allow for sorting downstream.
df_data$`Year collected` <- as.factor(df_data$`Year collected`)

# Create a new data frame sorted by the collection years in the column "Year collected" in ascending order. Then, omit any NAs from the data frame to make downstream analyses easier.
df_data_sorted <- df_data[order(df_data$`Year collected`), ] %>%
  na.omit()

# --- CREATING DATA FRAMES FOR EACH COLLECTION YEAR ---

# Create a list containing data frame subsets as elements, where each data frame subset corresponds to all entries for a collection year.
list_df_subsets <- split(df_data_sorted, df_data_sorted$`Year collected`)

# Create an empty vector that will contain the names of the data frames for each collection year.
vec_df_names <- c()

# Populate the empty vector by looping through all the names in the list of data frame subsets and adding each name to the empty vector.
for (i in names(list_df_subsets)) {
  vec_df_names <- append(vec_df_names, i)
}

# Add the prefix "df_" for each name in the vector for organizational purposes.
vec_df_names <- paste0('df_', vec_df_names, sep = '')

# Define a function that takes a data frame subset as input and filters it by:
  # 1) Setting the data in the column "Acc.No" as its row names.
  # 2) Deleting the columns "Acc.No," "Region," "Altitude," and "Year collected."
  # 3) Looping through the remaining columns and converting the data in each column into numeric format
filter_df_subsets <- function(df_subset) { # (L, 2015).
  row.names(df_subset) <- df_subset[, 1]
  df_subset <- df_subset[, -c(1, 2, 3, 4)]
  for (i in 1:ncol(df_subset)) {
    df_subset[, i] <- as.numeric(df_subset[, i])
  }
  return(df_subset)
}

# Create a new list containing filtered data frame subsets, filtering each data frame subset using the function defined above.
list_df_subsets_filtered <- lapply(list_df_subsets, function(df_subset) filter_df_subsets(df_subset)) # (mpalanco, 2015).

# Loop through each filtered data frame in our new list and assign an appropriate name to it according to its collection year
for (i in 1:length(list_df_subsets_filtered)) {
  assign(vec_df_names[i], list_df_subsets_filtered[[i]])
}

# --- FINDING THE MOST PROBABLE NUMBER OF CLUSTERS FOR EACH COLLECTION YEAR ---

# Find the most probable number of clusters in the data set by running the function "find.clusters" for each data frame subset. Unfortunately, "find.clusters" only works on data frames with a substantial number of entries (i.e., rows), so I can only run it on the data frame subsets for 1979 and 1986.
grp_1979 <- find.clusters(df_1979, n.pca = 200, method = 'kmeans', stat = 'BIC',  max.n.clust = 40)
grp_1986 <- find.clusters(df_1986, n.pca = 200, method = 'kmeans', stat = 'BIC',  max.n.clust = 40)
 
# When run, "find.clusters" will interactively ask the user to specify the number of retained PCs and the desired number of clusters (k). However, the user can bypass the interactive session entirely by explicitly providing the arguments "n.pca" and "n.clust," respectively (Jombart & Collins, 2015). Since I provided the argument "n.pca" but not "n.clust," I have to choose the number of clusters through the interactive session. In both interactive sessions, I chose k = 3, as it corresponds to the lowest BIC values.

# --- PERFORMING DAPC FOR EACH COLLECTION YEAR ---

# View the group memberships for each year to get a sense of their distributions.
grp_1979$grp
grp_1986$grp

# Using the corresponding group memberships identified above, perform DAPC on the data frame subsets for 1979 and 1986.
dapc_1979 <- dapc(df_1979, grp_1979$grp)
dapc_1986 <- dapc(df_1986, grp_1986$grp)

# Like "find.clusters," the function "dapc" also presents the user with an interactive session. I chose to retain 30 PCs and all eigenvalues (two in total for both data frame subsets) during both interactive sessions (Jombart & Collins, 2015).

# Plot scatter plots to visualize the clusters obtained by performing DAPC on the data frame subsets for 1979 and 1986.
scatter(dapc_1979)
scatter(dapc_1986)

# --- PERFORMING DAPC ON THE ENTIRE DATA SET ---

# Convert all data for SSR markers into numeric format.
df_data_sorted[, -c(1, 2, 3, 4)] <- as.numeric(unlist(df_data_sorted[, -c(1, 2, 3, 4)])) # (Zach, 2021).

# Run "find.clusters" on our original, sorted data frame. Specify n.clust = 3 to tell the function that we want it to find three clusters.
grp_main <- find.clusters(df_data_sorted[, -c(1, 2, 3, 4)], n.pca = 200, n.clust = 3, method = 'kmeans', stat = 'BIC',  max.n.clust = 40)

# Perform DAPC on our original, sorted data frame. Use the group memberships obtained from running "find.clusters."
dapc_main <- dapc(df_data_sorted[, -c(1, 2, 3, 4)], grp_main$grp)

# During the interactive session, I chose to retain 40 PCs and all eigenvalues (two in total).

# Plot a scatter plot to visualize the clusters obtained by performing DAPC on our original, sorted data frame. Customize the scatter plot so that it displays information in an easily understandable and aesthetically pleasing way.
colors <- c('#07099F', '#A40707', '#098D4B') # (Jombart & Collins, 2015).
scatter(dapc_main, col = colors, scree.da = TRUE, scree.pca = TRUE, posi.da = 'topright', posi.pca = 'topleft', bg = 'white', pch = 20, cex = 3) # (Jombart & Collins, 2015).

# --- REFERENCES ---
# Campoy, J. A., Lerigoleur-Balsemin, E., Christmann, H., Beauvieux, R., Girollet, N., Quero-Garcia, J., Dirlewanger, E., & Barreneche, T. (2016). Genetic diversity, linkage disequilibrium, population structure and construction of a core collection of Prunus avium L. landraces and bred cultivars. BMC Plant Biology, 16(1). https://doi.org/10.1186/s12870-016-0712-9
# Deperi, S. I., Tagliotti, M. E., Bedogni, M. C., Manrique-Carpintero, N. C., Coombs, J. E., Zhang, R., Douches, D. S., & Huarte, M. (2018). Discriminant analysis of principal components and pedigree assessment of genetic diversity and population structure in a tetraploid potato panel using SNPs. PLOS ONE, 13(3), e0194398. https://doi.org/10.1371/journal.pone.0194398
# Dido, A. A., Degefu, D. T., Assefa, E., Krishna, M. S. R., Singh, B. P., & Tesfaye, K. (2021). Spatial and temporal genetic variation in Ethiopian barley (Hordeum vulgare L.) landraces as revealed by simple sequence repeat (SSR) markers. Agriculture & Food Security, 10(1). https://doi.org/10.1186/s40066-021-00336-3
# Dido, A. A., Singh, B. J. K., Assefa, E., Krishna, M. S. R., Degefu, D., & Tesfaye, K. (2022). Spatial and temporal genetic variation in Ethiopian barley (Hordeum vulgare L.) landraces as revealed by simple sequence repeat (SSR) markers. Dryad. https://doi.org/10.5061/dryad.4tmpg4f8v
# Firke, S. (2023, February 2). janitor: Simple Tools for Examining and Cleaning Dirty Data. The Comprehensive R Archive Network. https://CRAN.R-project.org/package=janitor
# Jombart, T. (2008) adegenet: a R package for the multivariate analysis of genetic markers. Bioinformatics, 24(11), 1403-1405. https://doi.org/10.1093/bioinformatics/btn129
# Jombart, T., & Ahmed, I. (2011) adegenet 1.3-1: new tools for the analysis of genome-wide SNP data. Bioinformatics, 27(21), 3070-3071. https://doi.org/10.1093/bioinformatics/btr521
# Jombart, T., & Collins, C. (2015, June 23). A tutorial for Discriminant Analysis of Principal Components (DAPC) using adegenet 2.0.0. adegenet on the web. https://adegenet.r-forge.r-project.org/files/tutorial-dapc.pdf
# Jombart, T., Devillard, S., & Balloux, F. (2010). Discriminant analysis of principal components: a new method for the analysis of genetically structured populations. BMC Genetics, 11(1), 94. https://doi.org/10.1186/1471-2156-11-94
# L, P. [Pierre L]. (2015, August 17). use first row data as column names in r [Online forum post]. Stack Overflow. https://stackoverflow.com/questions/32054368/use-first-row-data-as-column-names-in-r
# Lee, K. J., Lee, J., Sebastin, R., Shin, M., Kim, S., Cho, G., & Hyun, D. Y. (2019). Genetic Diversity Assessed by Genotyping by Sequencing (GBS) in Watermelon Germplasm. Genes, 10(10), 822. https://doi.org/10.3390/genes10100822
# Milla, R. (2023). Phenotypic evolution of agricultural crops. Functional Ecology. https://doi.org/10.1111/1365-2435.14278
# Miller, J. D., Cullingham, C. I., & Peery, R. M. (2020). The influence of a priori grouping on inference of genetic clusters: simulation study and literature review of the DAPC method. Heredity, 125(5), 269–280. https://doi.org/10.1038/s41437-020-0348-2
# mpalanco [mpalanco]. (2015, August 17). use first row data as column names in r [Online forum post]. Stack Overflow. https://stackoverflow.com/questions/32054368/use-first-row-data-as-column-names-in-r
# RStudio Desktop. (n.d.). Posit. Retrieved April 14, 2023, from https://posit.co/download/rstudio-desktop/
# Schauberger, P., & Walker, A. (2023). openxlsx: Read, Write and Edit xlsx Files. The Comprehensive R Archive Network. https://CRAN.R-project.org/package=openxlsx
# Wickham, H., Averick, M., Bryan, J., Chang, W., McGowan, L. D., François, R., Grolemund, G., Hayes, A., Henry, L., Hester, J., Kuhn, M., Pedersen, T. L., Miller, E., Bache, S. M., Müller, K., Ooms, J., Robinson, D., Seidel, D. P., Spinu, V., Takahashi, K., Vaughan, D., Wilke, C., Woo, K., Yutani, H. (2019). Welcome to the tidyverse. Journal of Open Source Software, 4(43), 1686. https://doi.org/10.21105/joss.01686
# Wickham, H., François, R., Henry, L., Müller, K., & Vaughan, D. (2023, March 22). dplyr: A Grammar of Data Manipulation. The Comprehensive R Archive Network. https://CRAN.R-project.org/package=dplyr
# Zach. (2021, May 27). How to Fix: (list) object cannot be coerced to type ‘double’. Statology. https://www.statology.org/r-list-object-cannot-be-coerced-to-type-double/
# zek19 [zek19]. (2019, December 12). use first row data as column names in r [Online forum post]. Stack Overflow. https://stackoverflow.com/questions/32054368/use-first-row-data-as-column-names-in-r
