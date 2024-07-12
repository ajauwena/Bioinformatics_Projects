# Title                   : Performing Genomic Prediction to Predict Yields of Ontarian Barley Varieties
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
  # 1) Repeatedly computes 50 iterations of 5-fold cross-validation on SNP data from 54 barley varieties. Each instance is randomized via setting a new seed.
  # 2) Tests whether SNP density correlates with prediction accuracy for barley yields.
  # 3) Investigates whether old (i.e., pre-1990) varieties can predict new (i.e., post-1990) varieties.

# ----- Preparations -----

# Set the working directory to the one that has the dataset containing the called SNPs in "-101" format.

# Load the appropriate packages.
library(dplyr)
library(furrr) # Note that loading "furrr" changes "RNGkind()."
library(ggpmisc) # For adding regression line equations to plots
library(ggplot2)
library(rrBLUP)
library(scales)
library(tidyverse)

# Set the theme.
theme_set(theme_minimal(base_size = 14) +
            theme(legend.direction = "vertical",
                  legend.position = "top",
                  legend.key.width = unit(2, "cm")))

# ----- Reading in the Dataset -----

# Read in the dataset containing i) the SNP data in "-101" format and ii) the adjusted yield means per variety as a data frame. Use the first column as row names.
df_data <- read.csv("barley_merged_platypus_loci_imputed_subsetted_-101_mat_with_adjusted_yield_means_and_yfe.csv", row.names = 1)

# View the last ten column names of the data frame to see which columns contain metadata.
tail(colnames(df_data), n = 10)
# The last six columns contain metadata.

# Obtain the number of varieties in each year of entry.
df_varieties_per_yfe <- df_data %>%
  rownames_to_column(., var = "Variety") %>%
  distinct(Variety, Year_of_First_Entry) %>%
  arrange(Year_of_First_Entry) %>%
  count(Year_of_First_Entry) # (So, n.d.).

# Plot the number of varieties in each year of entry.
df_varieties_per_yfe %>%
  ggplot(aes(x = Year_of_First_Entry, y = n)) +
  geom_col() +
  coord_flip() +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Number of Varieties", y = "Year of Entry") # (So, n.d.).

# Create a genotype matrix containing only SNP markers by removing columns in the data frame that contain metadata. This matrix will be used to test the accuracy of the model's predictions.
mat_geno <- as.matrix(df_data[, 1:(length(df_data) - 6)]) # (rnso, 2014; So, n.d.).

# ----- Training the Model and Testing It on One Fold -----

# Prepare five folds (i.e., groups) to which all data points in the data frame will be assigned.
folds <- 5 # (So, n.d.).

# Define the total number of observations to sample from as the number of rows in the data frame.
N <- nrow(df_data) # (So, n.d.).

# Set a seed to ensure that the assignment of folds is identical from run to run.
set.seed(1000) # (So, n.d.).

# Assign data points randomly into five folds of roughly equal size. Altogether, these five folds make up one set.
sets <- sample(x = 1:folds, N, replace = TRUE) # (So, n.d.).

# View the size of each fold.
table(sets) # (So, n.d.).

# Create a vector of indices for the training set, which includes all folds except the first.
train_index <- which(sets != 1) # (So, n.d.).

# Copy the original data frame.
df_data_copy <- df_data # (So, n.d.).

# Mask the test values (i.e., the adjusted yield means) in the data frame copy.
df_data_copy[["Adjusted_Yield_Mean"]][-train_index] <- NA # (So, n.d.).

# Solve for marker effects.
marker_effects <- mixed.solve(df_data_copy[["Adjusted_Yield_Mean"]][train_index], Z = mat_geno[train_index, ], SE = FALSE) # (So, n.d.).

# Obtain the predicted test values by multiplying the genotype matrix with the marker effects and adding the mean.
pred <- mat_geno[-train_index, ] %*% marker_effects$u + c(marker_effects$beta) # (So, n.d.).

# Calculate the model's prediction accuracy.
cor(pred, df_data[["Adjusted_Yield_Mean"]][-train_index]) # (So, n.d.).
# The predicted values and the actual values for the testing set had a ~0.656 correlation.

# ----- Trying 5-fold Cross-Validation -----

# Create a data frame that will store the results of a 5-fold cross-validation procedure.
df_cv_trial <- c(1:folds) %>%
  # Create a function that performs cross-validation using each of the five folds.
  map_dfr(function(fold) {
    # Print a message specifying which fold is being cross-validated.
    print(paste("Cross-validating fold", fold))
    # Create a vector of indices for the training set, which includes all folds except the current one
    train_index <- which(sets != fold)
    # Copy the original data frame.
    df_data_copy <- df_data
    # Mask the test values in the data frame copy.
    df_data_copy[["Adjusted_Yield_Mean"]][-train_index] <- NA
    # Solve for marker effects.
    marker_effects <- mixed.solve(df_data_copy[["Adjusted_Yield_Mean"]][train_index], Z = mat_geno[train_index, ], SE = FALSE)
    # Obtain the predicted test values by multiplying the genotype matrix with the marker effects and adding the mean.
    pred <- mat_geno[-train_index, ] %*% marker_effects$u + c(marker_effects$beta)
    # For the current fold, create a data frame that contains i) the predicted test values, ii) the observed test values, iii) fold IDs, and iv) variety names.
    df_res <- data.frame(Predicted_Values = pred,
                         Observed_Values = df_data[["Adjusted_Yield_Mean"]][-train_index],
                         Fold_ID = fold) %>%
      rownames_to_column("Variety")
    # Return the data frame.
    return(df_res)
  }
    ) # (So, n.d.).

# Calculate the model's prediction accuracy for each fold in the data frame.
df_cv_trial %>%
  # Group the data frame by fold IDs for downstream operations (there are no observable effects).
  group_by(Fold_ID) %>%
  # Calculate the correlations between the predicted and observed values.
  summarize(Correlation = cor(Predicted_Values, Observed_Values)) %>%
  # Remove the fold ID grouping from the data frame (there are no observable effects).
  ungroup() %>%
  # Calculate the means and standard deviations of the correlations.
  mutate(Mean_Correlation = mean(Correlation), SD = sqrt(var(Correlation))) # (So, n.d.).
# Training the model on folds 1, 2, 3, and 5 and testing it on fold 4 yielded the highest prediction accuracy (0.658) on adjusted yield means. The model also exhibits a mean correlation of 0.534 with a standard deviation of 0.154.

# Visualize the prediction accuracy of each fold.
df_cv_trial %>%
  mutate(Fold_ID = as.factor(Fold_ID)) %>%
  ggplot(aes(x = Predicted_Values, y = Observed_Values)) +
  geom_point(aes(colour = Fold_ID), show.legend = FALSE, size = 2) +
  facet_wrap(~ Fold_ID) +
  scale_fill_brewer(palette = "Set1") + 
  labs(x = "Predicted Values", y = "Observed Values") # (So, n.d.).

# ----- Creating Functions for Downstream Use -----

# Create a function that assigns observations in the main dataset into sets (default 10). Each assignment of observations into a set is called an "iteration." In turn, each set contains folds (default 5) that contains distinct observations.
# The function takes in the following as input:
  # 1) The main dataset to use.
  # 2) The number of folds.
  # 3) The number of iterations.
  # 4) The seed, which ensures that the assignment of sets is identical from run to run. Change seeds to change the assignment of sets.
set_assigner <- function(main_data, folds = 5, iterations = 10, seed) {
  # Define the total number of observations to sample from as the number of rows in the main dataset.
  N <- nrow(main_data)
  # Create a vector containing the folds to which observations will be assigned.
  sampling_folds <- rep(1:folds, length.out = nrow(df_data))
  # Sample for all iterations in parallel.
  future_map_dfr(c(1:iterations), .options = furrr_options(seed = seed), function(i) {
    # Print a message specifying the current iteration.
    print(paste("Iteration", i))
    # Randomly assign observations in the iteration to folds to create a unique set.
    sets <- sample(sampling_folds)
    # Print the current set to see the fold's contents.
    print(table(sets))
    # Create a data frame that contains i) fold IDs and ii) variety names.
    df_folds <- data.frame(Fold_ID = sets) %>%
      rownames_to_column("Variety")
    # Create a column in the data frame specifying the iteration to which each observation belongs.
    df_folds$Iteration <- i
    # Return the data frame as a tibble.
    return(as_tibble(df_folds))
  }, .progress = TRUE)
} # (Smith, 2023).

# Create a function that performs repeated k-fold cross-validation on the main dataset.
# The function takes in the following as input:
  # 1) The main dataset to use.
  # 2) The set dataset, which was outputted as a tibble by the function "set_assigner."
  # 3) The genotype matrix (containing only SNP markers) using which the model will predict the test values.
  # 4) The column in the main dataset containing the test values on which the model's performance will be tested.
repeated_cross_validator <- function(main_data, set_data, genotype_matrix, test_values) {
  # Set the number of iterations.
  num_iterations <- length(unique(set_data$Iteration))
  # Set the number of folds.
  num_folds  <- length(unique(set_data$Fold_ID))
  # Remove any quotes from the test values.
  y <- noquote(test_values)
  # Map the unquoted test values to the main dataset.
  main_data[[y]] <- main_data[[test_values]]
  # Perform cross-validation for each iteration.
  group_iteration <- c(1:num_iterations) %>%
    map_dfr(function(iteration) {
      # Print a message specifying the current iteration.
      print(paste("Iteration", iteration))
      # Select the rows in the set dataset that correspond to the current iteration.
      set_data_iteration <- set_data[set_data$Iteration == iteration, ]
      # Perform cross-validation for each fold in the current iteration, then output the result to a data frame.
      df_cv_folds <- c(1:num_folds) %>%
        map(function(fold) {
          # Print a message specifying which fold is being cross-validated.
          print(paste("Cross-validating fold", fold))
          # Create a vector of indices for the training set, which includes all folds except the current one.
          train_index <- which(set_data_iteration$Fold_ID != fold)
          # Copy the original data frame.
          main_data_copy <- main_data
          # Mask the test values in the data frame copy.
          main_data_copy[[y]][-train_index] <- NA
          # Solve for marker effects.
          marker_effects <- mixed.solve(main_data_copy[[y]][train_index], Z = genotype_matrix[train_index, ], SE = FALSE)
          # Obtain the predicted test values by multiplying the genotype matrix with the marker effects and adding the mean.
          pred <- genotype_matrix[-train_index, ] %*% marker_effects$u + c(marker_effects$beta)
          # Create a data frame that contains i) the predicted test values, ii) the observed test values, iii) fold IDs, and iv) variety names.
          df_res <- data.frame(Predicted_Values = pred,
                               Observed_Values = df_data[["Adjusted_Yield_Mean"]][-train_index],
                               Fold_ID = fold) %>%
            rownames_to_column("Variety")
          # Return the data frame.
          return(df_res)
        }) %>%
        # Bind the rows of all the returned data frames, where each data frame contains the cross-validation results from an iteration.
        bind_rows()
      # Create a column in the data frame containing the current iteration.
      df_cv_folds$Iteration <- iteration
      # Return the data frame as a tibble.
      return(as_tibble(df_cv_folds))
  })
} # (Smith, 2023).

# Create a function that evaluates the model's prediction accuracy (PA) after performing repeated k-fold cross-validation.
# The function takes in the following as input:
  # 1) The genomic prediction (GP) dataset, which was outputted as a tibble by the function "repeated_cross_validator."
model_pa_evaluator <- function(gp_data) {
  
  # Create a data frame that contains the mean PA per iteration.
  df_mean_correlation_per_iteration <- gp_data %>%
    # Group the GP dataset by iteration, then by folds.
    group_by(Iteration, Fold_ID) %>%
    # Calculate the PA for each fold in each iteration.
    summarize(PA_per_Fold = cor(Predicted_Values, Observed_Values)) %>%
    # Group the GP dataset again by iteration.
    group_by(Iteration) %>%
    # Calculate the PA per iteration.
    summarize(PA_per_Iteration = mean(PA_per_Fold), n = n())
  
  # Create a data frame that contains the overall PA
  df_overall_correlation <- df_mean_correlation_per_iteration %>%
    # Remove the iteration grouping from the GP dataset.
    ungroup() %>%
    # Calculate the mean and standard deviation of the overall PA.
    summarize(PA_Overall = mean(PA_per_Iteration), n = n(), stdev = sqrt(var(PA_per_Iteration)))
  
  # Create a data frame that contains i) the mean PA per fold, ii) the mean PA per iteration, and iii) the mean and standard deviation of the overall PA.
  df_correlations_per_fold <- gp_data %>%
    # Group the GP dataset by iteration, then by folds.
    group_by(Iteration, Fold_ID) %>%
    # Calculate the model's prediction accuracy for each fold in each iteration.
    summarize(PA_per_Fold = cor(Predicted_Values, Observed_Values)) %>%
    # Add columns corresponding to i) the mean PA per iteration and ii) the mean and standard deviation of the overall PA.
    mutate(PA_per_Iteration = df_mean_correlation_per_iteration$PA_per_Iteration[Iteration],
           PA_Overall = df_overall_correlation$PA_Overall)
  
  # Print the first two data frames.
  print(df_mean_correlation_per_iteration)
  print(df_overall_correlation)
  
  # Return the last data frame.
  return(df_correlations_per_fold)
} # (Smith, 2023).

# ----- Performing Repeated 5-fold Cross-Validation: Run 1 -----

# Assign observations in the main dataset into distinct sets. Repeat for 50 iterations.
df_sets_actual_1 <- set_assigner(main_data = df_data, folds = 5, iterations = 50, seed = set.seed(1000))

# Perform repeated k-fold cross-validation on the main dataset.
df_cv_actual_1 <- repeated_cross_validator(main_data = df_data, set_data = df_sets_actual_1, genotype_matrix = mat_geno, test_values = "Adjusted_Yield_Mean")

# Obtain the average predicted yields per variety.
df_average_predicted_values_per_variety_1 <- df_cv_actual_1 %>%
  group_by(Variety) %>%
  summarize(Mean_Predicted_Values = mean(Predicted_Values))

# Append the average predicted yields per variety to the data frame containing the predicted and observed yields.
df_cv_actual_1_averaged <- left_join(df_cv_actual_1, df_average_predicted_values_per_variety_1)

# Plot the predicted values outputted by the model against the observed values.
df_cv_actual_1_averaged %>%
  ggplot(aes(x = Observed_Values, y = Mean_Predicted_Values, linetype = "scatterplot", colour = "scatterplot"), linewidth = 2) +
  geom_point(size = 2, color = "#99ccff") +
  geom_smooth(mapping = aes(x = Observed_Values, y = Mean_Predicted_Values), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Observed Adjusted Yield Means (kg/ha)", y = "Average Predicted Adjusted Yield Means (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("R2"), label.y = 0.9) +
  scale_linetype_manual(values = "solid", labels = "Line of Best Fit", name = "") +
  scale_colour_manual(values = "#000099", labels = "Line of Best Fit", name = "")

# Evaluate the overall PA of the first model.
df_evaluation_actual_1 <- model_pa_evaluator(df_cv_actual_1)
# The overall PA is 0.557.

# ----- Performing Repeated 5-fold Cross-Validation: Run 2 -----

# Assign observations in the main dataset into distinct sets. Repeat for 50 iterations.
df_sets_actual_2 <- set_assigner(main_data = df_data, folds = 5, iterations = 50, seed = set.seed(2000))

# Perform repeated k-fold cross-validation on the main dataset.
df_cv_actual_2 <- repeated_cross_validator(main_data = df_data, set_data = df_sets_actual_2, genotype_matrix = mat_geno, test_values = "Adjusted_Yield_Mean")

# Obtain the average predicted yields per variety.
df_average_predicted_values_per_variety_2 <- df_cv_actual_2 %>%
  group_by(Variety) %>%
  summarize(Mean_Predicted_Values = mean(Predicted_Values))

# Append the average predicted yields per variety to the data frame containing the predicted and observed yields.
df_cv_actual_2_averaged <- left_join(df_cv_actual_2, df_average_predicted_values_per_variety_2)

# Plot the predicted values outputted by the model against the observed values.
df_cv_actual_2_averaged %>%
  ggplot(aes(x = Observed_Values, y = Mean_Predicted_Values, linetype = "scatterplot", colour = "scatterplot"), linewidth = 2) +
  geom_point(size = 2, color = "#99ccff") +
  geom_smooth(mapping = aes(x = Observed_Values, y = Mean_Predicted_Values), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Observed Adjusted Yield Means (kg/ha)", y = "Average Predicted Adjusted Yield Means (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("R2"), label.y = 0.9) +
  scale_linetype_manual(values = "solid", labels = "Line of Best Fit", name = "") +
  scale_colour_manual(values = "#000099", labels = "Line of Best Fit", name = "")

# Evaluate the overall PA of the second model.
df_evaluation_actual_2 <- model_pa_evaluator(df_cv_actual_2)
# The overall PA is 0.570.

# ----- Performing Repeated 5-fold Cross-Validation: Run 3 -----

# Assign observations in the main dataset into distinct sets. Repeat for 50 iterations.
df_sets_actual_3 <- set_assigner(main_data = df_data, folds = 5, iterations = 50, seed = set.seed(3000))

# Perform repeated k-fold cross-validation on the main dataset.
df_cv_actual_3 <- repeated_cross_validator(main_data = df_data, set_data = df_sets_actual_3, genotype_matrix = mat_geno, test_values = "Adjusted_Yield_Mean")

# Obtain the average predicted yields per variety.
df_average_predicted_values_per_variety_3 <- df_cv_actual_3 %>%
  group_by(Variety) %>%
  summarize(Mean_Predicted_Values = mean(Predicted_Values))

# Append the average predicted yields per variety to the data frame containing the predicted and observed yields.
df_cv_actual_3_averaged <- left_join(df_cv_actual_3, df_average_predicted_values_per_variety_3)

# Plot the predicted values outputted by the model against the observed values.
df_cv_actual_3_averaged %>%
  ggplot(aes(x = Observed_Values, y = Mean_Predicted_Values, linetype = "scatterplot", colour = "scatterplot"), linewidth = 2) +
  geom_point(size = 2, color = "#99ccff") +
  geom_smooth(mapping = aes(x = Observed_Values, y = Mean_Predicted_Values), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Observed Adjusted Yield Means (kg/ha)", y = "Average Predicted Adjusted Yield Means (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("R2"), label.y = 0.9) +
  scale_linetype_manual(values = "solid", labels = "Line of Best Fit", name = "") +
  scale_colour_manual(values = "#000099", labels = "Line of Best Fit", name = "")

# Evaluate the overall PA of the third model.
df_evaluation_actual_3 <- model_pa_evaluator(df_cv_actual_3)
# The overall PA is 0.543.

# ----- Visualizing the Combined Prediction Accuracy of All Three Models -----

# Merge the data frames containing the predicted and observed values for all three models.
df_cv_actual_merged <- full_join(df_cv_actual_1, df_cv_actual_2) %>%
  full_join(df_cv_actual_3)

# Plot the predicted values outputted by all three models against the observed values.
df_cv_actual_merged %>%
  ggplot(aes(x = Observed_Values, y = Predicted_Values, linetype = "scatterplot", colour = "scatterplot"), linewidth = 2) +
  geom_point(size = 3, color = "#999999") +
  geom_smooth(mapping = aes(x = Observed_Values, y = Predicted_Values), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Observed Adjusted Yield Means (kg/ha)", y = "Predicted Adjusted Yield Means (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("R2"), label.y = 0.9) +
  scale_linetype_manual(values = "solid", labels = "Line of Best Fit", name = "") +
  scale_colour_manual(values = "#990000", labels = "Line of Best Fit", name = "")

# Obtain the average predicted yields per variety.
df_average_predicted_values_per_variety_merged <- df_cv_actual_merged %>%
  group_by(Variety) %>%
  summarize(Mean_Predicted_Values = mean(Predicted_Values))

# Append the average predicted yields per variety to the merged data frame containing the predicted and observed yields.
df_cv_actual_merged_averaged <- left_join(df_cv_actual_merged, df_average_predicted_values_per_variety_merged)

# Merge the data frames containing the PA across 50 iterations for all three models.
df_evaluation_actual_merged <- full_join(df_evaluation_actual_1, df_evaluation_actual_2) %>%
  full_join(df_evaluation_actual_3)

# Calculate the mean correlation coefficient (PA) between the predicted and observed values.
r_merged_averaged <- mean(df_evaluation_actual_merged$PA_per_Iteration)
r_merged_averaged
# The mean correlation coefficient is ~0.557.

# Calculate the standard deviation of the PA across 50 iterations for all three models.
sd_merged_averaged <- sd(df_evaluation_actual_merged$PA_per_Iteration)
sd_merged_averaged
# The standard deviation is ~0.062.

# Plot the average predicted values outputted by all three models against the observed values.
df_cv_actual_merged_averaged %>%
  ggplot(aes(x = Observed_Values, y = Mean_Predicted_Values, linetype = "scatterplot", colour = "scatterplot"), linewidth = 2) +
  geom_point(size = 3, color = "#999999") +
  geom_smooth(mapping = aes(x = Observed_Values, y = Mean_Predicted_Values), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Observed Adjusted Yield Means (kg/ha)", y = "Average Predicted Adjusted Yield Means (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  annotate("text", x = min(df_cv_actual_merged_averaged$Mean_Predicted_Values), y = max(df_cv_actual_merged_averaged$Mean_Predicted_Values), label = paste("r = ", round(r_merged_averaged, 3)), colour = "#990000", hjust = 5.56, vjust = 3.5) +
  scale_linetype_manual(values = "solid", labels = "Line of Best Fit", name = "") +
  scale_colour_manual(values = "#990000", labels = "Line of Best Fit", name = "")

# Plot the overall PA across 50 iterations for all three models.
df_evaluation_actual_merged %>%
  mutate(Population_Size = as.factor(54)) %>%
  ggplot(aes(x = Population_Size, y = PA_per_Iteration)) +
  geom_jitter(width = 0.1, size = 1, color = "#999999", alpha = 0.7) +
  geom_boxplot(width = 0.2, size = 1, color = "#990000", alpha = 0.7, show.legend = FALSE) +
  labs(x = "Population Size", y = "Prediction Accuracy per Iteration") # (So, n.d.).

# ----- Testing Whether Old Varieties Can Predict New Varieties (and Vice Versa) -----

# Create a data frame that distinguishes between old (i.e., pre-1990) and new (i.e., post-1990) varieties.
df_old_vs_new_varieties <- df_data[c("Year_of_First_Entry")] %>%
  # Add a column that distinguishes between old and new varieties.
  mutate(Generation = case_when(Year_of_First_Entry >= 1990 ~ "Post-1990",
                                Year_of_First_Entry < 1990 ~ "Pre-1990")) # (So, n.d.).

# Count the number of old vs. new varieties.
table(df_old_vs_new_varieties["Generation"]) # (So, n.d.).
# There are 25 old varieties and 29 new varieties.

# Create a vector of indices for old and new varieties.
old_varieties_index <- which(df_old_vs_new_varieties$Generation == "Pre-1990") # (So, n.d.).
new_varieties_index <- which(df_old_vs_new_varieties$Generation == "Post-1990") # (So, n.d.).

# Copy the original data frame for predicting new varieties.
df_data_copy_pred_new_varieties <- df_data # (So, n.d.).

# Mask the test values in the data frame copy.
df_data_copy_pred_new_varieties$Adjusted_Yield_Mean[which(df_old_vs_new_varieties$Generation == "Post-1990")] <- NA # (So, n.d.).

# Solve for marker effects.
marker_effects_pred_new_varieties <- mixed.solve(df_data_copy_pred_new_varieties$Adjusted_Yield_Mean[old_varieties_index], Z = mat_geno[old_varieties_index, ], SE = FALSE) # (So, n.d.).

# Obtain the predicted test values for the new varieties by multiplying the genotype matrix with the marker effects and adding the mean.
test_pred_new_varieties <- mat_geno[new_varieties_index, ] %*% marker_effects_pred_new_varieties$u + c(marker_effects_pred_new_varieties$beta) # (So, n.d.).

# Calculate the model's prediction accuracy.
r_pred_new_varieties <- cor(test_pred_new_varieties, df_data$Adjusted_Yield_Mean[new_varieties_index]) # (So, n.d.).
r_pred_new_varieties
# Old varieties do not seem to predict new varieties, as seen from the low (~-0.122) correlation.

# Create a data frame containing the predicted and observed test values that result from trying to use old varieties to predict new varieties.
df_ggplot_pred_new_varieties <- tibble(Predicted_Values = test_pred_new_varieties, Observed_Values = df_data$Adjusted_Yield_Mean[new_varieties_index], Generation = "Post-1990") # (So, n.d.).

# Visualize the results of trying to use old varieties to predict new varieties.
df_ggplot_pred_new_varieties %>%
  ggplot(aes(x = Predicted_Values, y = Observed_Values, linetype = "scatterplot", colour = "scatterplot"), linewidth = 2) +
  geom_point(size = 3, color = "#990000") +
  geom_smooth(mapping = aes(x = Predicted_Values, y = Observed_Values), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Observed Adjusted Yield Means (kg/ha)", y = "Predicted Adjusted Yield Means (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  annotate("text", x = min(df_ggplot_pred_new_varieties$Predicted_Values), y = max(df_ggplot_pred_new_varieties$Predicted_Values), label = paste("r = ", round(r_pred_new_varieties, 3)), colour = "#990000", hjust = -0.08, vjust = -28) +
  scale_linetype_manual(values = "solid", labels = "Line of Best Fit", name = "") +
  scale_colour_manual(values = "#990000", labels = "Line of Best Fit", name = "") # (So, n.d.).

# Copy the original data frame for predicting old varieties.
df_data_copy_pred_old_varieties <- df_data # (So, n.d.).

# Mask the test values in the data frame copy.
df_data_copy_pred_old_varieties$Adjusted_Yield_Mean[which(df_old_vs_new_varieties$Generation == "Pre-1990")] <- NA # (So, n.d.).

# Solve for marker effects.
marker_effects_pred_old_varieties <- mixed.solve(df_data_copy_pred_old_varieties$Adjusted_Yield_Mean[new_varieties_index], Z = mat_geno[new_varieties_index, ], SE = FALSE) # (So, n.d.).

# Obtain the predicted test values for the old varieties by multiplying the genotype matrix with the marker effects and adding the mean.
test_pred_old_varieties <- mat_geno[old_varieties_index, ] %*% marker_effects_pred_old_varieties$u + c(marker_effects_pred_old_varieties$beta) # (So, n.d.).

# Calculate the model's prediction accuracy.
r_pred_old_varieties <- cor(test_pred_old_varieties, df_data$Adjusted_Yield_Mean[old_varieties_index]) # (So, n.d.).
r_pred_old_varieties
# New varieties do not seem to predict old varieties, as seen from the low (~0.016) correlation.

# Create a data frame containing the predicted and observed test values that result from trying to use new varieties to predict old varieties.
df_ggplot_pred_old_varieties <- tibble(Predicted_Values = test_pred_old_varieties, Observed_Values = df_data$Adjusted_Yield_Mean[old_varieties_index], Generation = "Pre-1990") # (So, n.d.).

# Visualize the results of trying to use new varieties to predict old varieties.
df_ggplot_pred_old_varieties %>%
  ggplot(aes(x = Predicted_Values, y = Observed_Values, linetype = "scatterplot", colour = "scatterplot"), linewidth = 2) +
  geom_point(size = 3, color = "#000099") +
  geom_smooth(mapping = aes(x = Predicted_Values, y = Observed_Values), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Observed Adjusted Yield Means (kg/ha)", y = "Predicted Adjusted Yield Means (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  annotate("text", x = min(df_ggplot_pred_old_varieties$Predicted_Values), y = max(df_ggplot_pred_old_varieties$Predicted_Values), label = paste("r = ", round(r_pred_old_varieties, 3)), colour = "#000099", hjust = -0.09, vjust = 3.5) +
  scale_linetype_manual(values = "solid", labels = "Line of Best Fit", name = "") +
  scale_colour_manual(values = "#000099", labels = "Line of Best Fit", name = "") # (So, n.d.).

# ----- References -----

# rnso. [rnso]. (2014, October 21). Drop last 5 columns from a dataframe without knowing specific number [Online forum post]. Stack Overflow. https://stackoverflow.com/questions/26483643/drop-last-5-columns-from-a-dataframe-without-knowing-specific-number
# Smith, A. (2023, June 7). Functions for genomic prediction within and across breeding cycles [R script]. The R Foundation.
# So, D. (n.d.). Chapter 4 - Genomic Selection of OCCC Winter Wheat Cultivars using RRBLUP [R script]. The R Foundation.
