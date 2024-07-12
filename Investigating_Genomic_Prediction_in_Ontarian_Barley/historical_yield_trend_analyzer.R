# Title                   : Analyzing the Historical Yield Trends of Ontarian Barley Varieties
# Author                  : Abelhard Jauwena
# Student Number          : 1040413
# Course Code             : BINF*6999
# Course Name             : Bioinformatics Master's Project
# Submission Date         : December 15, 2023
# University Advisors     :
  # Informatics Expertise : Dr. Yan Yan, School of Computer Science.
  # Biological Expertise  : Dr. Lewis Lukens, Department of Plant Agriculture

# ----- Instructions -----

# This script elucidates the historical yield trends of Ontarian barley varieties, specifically the overall trend, genetic trend, and environmental trend. Yield data was obtained from the annual barley performance trials conducted by the Ontario Cereal Crop Committee (OCCC) (So et al., 2022).

# ----- Loading Packages -----

# Set your working directory to the one that has the dataset containing historical barley yields.

# Load the appropriate packages.
library(BiocManager)
library(broom)
library(DescTools) # Contains the "Winsorize()" function.
library(dplyr) # Used for calculating yield means per year and includes the "arrange()" function.
library(emmeans) # Contains the "emmeans()" function for calculating the adjusted year and entry means.
library(ggpmisc) # For adding regression line equations to plots
library(lme4)
library(naniar)
library(patchwork)
library(RColorBrewer)
library(readxl) # For reading in the dataset.
library(tidyverse)

# ----- Theme Settings -----

# Set the theme.
theme_set(theme_bw(base_size = 14) +
            theme(legend.direction = "vertical",
                  legend.position = "top",
                  legend.key.width = unit(2, "cm")))

# ----- Exploring the Dataset -----

# Read in the dataset as a tibble.
df_data <- read_excel("data_AllOCCCBarley_copy.xlsx") %>%
  as_tibble()

# View the dataset's dimensions.
dim(df_data)

# View the dataset's column names (variables) and data types
names(df_data)
str(df_data)
# The dataset consists of the following variables:
  # "Variety" containing characters.
  # "Year" containing numeric data.
  # "Yield" containing numeric data.
  # "Location" containing characters.
  # "Area" containing numeric data.
  # "Category" containing characters.

# Check for any missing entries in each column.
miss_var_summary(df_data)
# 87 entries were missing in the column "Yield," which corresponds to only ~0.559% of the total entries. Therefore, all missing entries can be safely omitted.

# Check the unique barley varieties included in the dataset.
sort(unique(df_data$Variety))
length(sort(unique(df_data$Variety)))
# The dataset includes 372 unique barley varieties.

# Check the unique years in which data were entered into the trial.
sort(unique(df_data$Year))
# Data were entered into the trial from 1958 to 2021.

# Check if there are any gap years.
2021 - 1958
length(sort(unique(df_data$Year)))
# There are two gap years between 1958 and 2021.

# Check the unique locations included in the dataset.
sort(unique(df_data$Location))
length(sort(unique(df_data$Location)))
# The dataset includes 199 unique locations.

# Check the unique areas included in the dataset, where an area consists of multiple locations.
sort(unique(df_data$Area))
length(sort(unique(df_data$Area)))
# The dataset includes six unique areas.

# Count the number of entries per variety.
df_data_variety_count <- df_data %>%
  # Omit all missing entries.
  na.omit() %>%
  # Only retain rows corresponding to unique year-variety combinations to prevent duplicates.
  distinct(Year, Variety) %>%
  # Count the number of entries per variety, sorting from most many to fewest.
  count(Variety, sort = TRUE)

# Count the total number of varieties.
length(df_data_variety_count$Variety)
# There are 371 varieties in total.

# List all low-count varieties, which are varieties that have a cumulative occurrence of less than two across the years. The cutoff value was set to two for several reasons:
  # 1) To retain as many entries as possible.
  # 2) To ensure that the entries remain as connected as possible across the years, allowing for comparison.
vec_low_count_varieties <- df_data_variety_count$Variety[(df_data_variety_count$n) < 2]

# Count the number of low-count varieties and their proportion relative to all the other varieties.
length(vec_low_count_varieties)
(length(vec_low_count_varieties) / length(df_data_variety_count$Variety)) * 100
# There are 156 low-count varieties, which comprise ~42.049% of the total varieties. Later, these varieties will be omitted when filtering the dataset.

# Obtain the year of first entry (YFE) for each variety, where applicable.
df_data_yfe_per_variety <- df_data %>%
  # Retain only rows corresponding to unique year-variety combinations ("distinct()" is case-sensitive).
  distinct(Year, Variety) %>%
  # Arrange the rows by year (from oldest to most recent), then by variety (in alphabetical order).
  arrange(Year, Variety) %>%
  # Group the tibble by variety for downstream operations (there are no observable effects).
  group_by(Variety) %>%
  # For each variety, only retain the first row, which corresponds to its YFE.
  filter(row_number() == 1) %>%
  # Remove the variety grouping from the tibble (there are no observable effects).
  ungroup() %>%
  # Add a column containing the YFE for each variety as integers. I will use this column to analyze the genetic trend.
  mutate(Year_of_First_Entry = as.integer(Year)) %>%
  # Remove the original column containing years.
  select(-Year)

# For each year, count the number of varieties that entered the trial for the first time.
df_data_variety_count_per_year <- df_data_yfe_per_variety %>%
  count(Year_of_First_Entry)

# ----- Filtering the Dataset -----

# Filter the dataset for use in downstream analyses.
df_data_filtered <- df_data %>%
  # Omit all missing entries.
  na.omit() %>%
  # Omit all low-count varieties.
  filter(!Variety %in% vec_low_count_varieties) %>%
  # Add the YFE of each variety.
  left_join(df_data_yfe_per_variety) %>%
  # Add a new column "Calendar_Year," which contains entries from the column "Year" as integers. These entries will be used to analyze the environmental trend. Then, convert these entries into factors for use in analyzing historical barley performance in trials.
  mutate(Calendar_Year = as.integer(Year), Year = as.factor(Year)) %>%
  # Sort the entries by years in ascending order.
  arrange(Year)
# After filtering, the dataset includes the following columns:
  # 1) "Variety" – The name of the ith variety.
  # 2) "Year" – The year in which the variety entered the trial.
  # 3) "Yield" – The yield of the variety in the kth year.
  # 4) "Location" – The jth location in which the ith variety was grown.
  # 5) "Area" – The area in which the yield data for the ith variety was taken.
  # 6) "Category" – The category in which the ith variety belongs, which is always "Barley."

# Count the number of varieties included in the filtered dataset.
df_data_filtered %>% distinct(Variety) %>%
  count()
# The filtered dataset contains 215 varieties in total.

# Save the filtered dataset as a .csv file.
write_csv(df_data_filtered, "data_allocccbarley_copy_filtered.csv")

# ----- Obtaining the Overall Trend -----

# Linearly regress yields by years, where years are treated as factor levels.
fit_lm_overall <- lm(Yield ~ Year, df_data_filtered)

# Calculate and view the adjusted yield means per year.
df_group_adjusted_means_overall <- emmeans(fit_lm_overall, "Year", rg.limit = 15000) %>%
  broom::tidy()
head(df_group_adjusted_means_overall)

# Convert the data in the column "Year" into integers.
df_group_adjusted_means_overall$Year <- as.integer(df_group_adjusted_means_overall$Year)

# Plot the adjusted yield means against years to visualize the overall trend.
ggplot(data = df_group_adjusted_means_overall, aes(x = Year, y = estimate, linetype = "overall", colour = "overall"), linewidth = 2) +
  geom_point(size = 3) +
  geom_smooth(data = df_group_adjusted_means_overall, mapping = aes(x = Year, y = estimate), method = "lm", se = FALSE, na.rm = TRUE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90)) +
  labs(x = "Year of Trial", y = "Adjusted Yield Mean (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("P"), label.y = 0.9) +
  scale_x_continuous(breaks = seq(from = 1958, to = 2021, by = 1)) +
  scale_y_continuous(limits = c(2500, 5500)) +
  scale_linetype_manual(values = "solid", labels = "Overall Trend", name = "") +
  scale_colour_manual(values = "#000000", labels = "Overall Trend", name = "")
# The rate of yield increase is estimated to be ~33.7 kg/ha/yr and is also significant, with p < 0.001.

# Create a new data frame containing the adjusted yield means from 1958 to 1989.
df_group_adjusted_means_overall_1 <- df_group_adjusted_means_overall %>% filter(Year <= 1989)

# Obtain the average adjusted yield mean from 1958 to 1989.
mean(df_group_adjusted_means_overall_1$estimate)
# The average adjusted yield mean from 1958 to 1989 is 3196.896 kg/ha.

# Create a new data frame containing the adjusted yield means from 1990 to 2021.
df_group_adjusted_means_overall_2 <- df_group_adjusted_means_overall %>% filter(Year > 1989)

# Obtain the average adjusted yield mean from 1990 to 2021.
mean(df_group_adjusted_means_overall_2$estimate)
# The average adjusted yield mean from 1990 to 2021 is 4461.418 kg/ha.

# ----- Estimating Variance Components for the Overall Trend -----

# Laidig et al. (2008) proposed a linear mixed model (LMM) that predicts crop yields in multi-year variety data trials based on genetic and environmental factors, which include genotype (i.e., variety), location, and year. This LMM will be used to estimate variance components for the overall trend.

# The LMM was given as "yijk = μ + Gi + Lj + Yk + (LY)jk + (GL)ij + (GY)ik + (GLY)ijk," where:
  # yijk – The mean yield of the ith variety in the jth location and kth year,
  # μ – The mean overall yield,
  # Gi – The main effect of the ith variety,
  # Lj – The main effect of the jth location,
  # Yk – The main effect of the kth year,
  # (LY)jk – The jkth location × year interaction effect,
  # (GL)ij – The ijth variety × location interaction effect,
  # (GY)ik – The ikth variety × year interaction effect, and...
  # (GLY)ijk – A residual comprising the variety × location × year interaction effect and the error of a mean (Piepho et al., 2014).

# Create a baseline LMM using the above model.
lmm_baseline <- formula(Yield ~ (1|Variety) + Year_of_First_Entry + (1|Location) + (1|Year) + Calendar_Year + (1|Location:Year) + (1|Variety:Location) + (1|Variety:Year)) # (So et al., 2022; So, n.d.).
# Terms preceded by "(1|)" are treated as random effects; else, they are treated as fixed effects (So, n.d.).

# Fit the baseline LMM to the filtered dataset.
fit_lmm_baseline <- lmer(lmm_baseline, data = df_data_filtered)

# View the summary of the fitted baseline LMM.
summary(fit_lmm_baseline) # (So, n.d.).

# Plot the residuals of the fitted baseline LMM.
plot(fit_lmm_baseline, main = "Residual plot of the baseline linear mixed model", xlab = "Yield", ylab = "Residual Values") # (So, n.d.).

# Estimate the variance components of the fitted baseline LMM, which includes the total variation summed across all random effects and the proportion and percentage of variation for each random effect.
df_varcorr_fit_lmm_baseline <- data.frame(VarCorr(fit_lmm_baseline)) %>%
  mutate(Total_Variation = sum(vcov),
         Proportion_of_Variation = round(vcov / Total_Variation, 3),
         Percentage_of_Variation = (Proportion_of_Variation * 100)) # (So, n.d.).

# View the estimated variance components.
df_varcorr_fit_lmm_baseline
# The most important variance components, defined as the components that accounted for most of yield variations, are listed below in descending order:
  # 1) Location × Year interaction (57%).
  # 2) Location (14.6%).
  # 3) Residual (13.8%).
  # 4) Year (10.5%).
  # 5) Variety (2.9%).
  # 6) Variety × Year interaction (0.9%).
  # 7) Variety × Location interaction (0.2%).

# Save the estimated variance components as a .csv file.
write_csv(df_varcorr_fit_lmm_baseline, "estimated_variance_components_overall_historical_barley_yield_trend.csv")

# ----- Obtaining the Genetic Trend -----

# The model "Gi = βri + Hi" can be used to model the genetic trend, where:
  # Gi – The main effect of the ith variety,
  # β – A fixed regression coefficient for the genetic trend,
  # ri – The YFE, and...
  # Hi – The random deviation of Gi from the genetic trend line, assumed to follow a normal distribution with a mean of zero and a variance of σ2H (Piepho et al., 2014).

# To do this, treat 1) "Year_of_First_Entry" as a categorical effect (i.e., convert it into factors) and 2) "Year" as a fixed effect to remove the environmental effect (So, n.d.).

# Create another LMM to model the genetic trend.
lmm_genetic_trend <- formula(Yield ~ (1|Variety) + as.factor(Year_of_First_Entry) + (1|Location) + Year + (1|Location:Year) + (1|Variety:Location) + (1|Variety:Year)) # (So, n.d.).

# Fit the LMM for the genetic trend to the filtered dataset.
fit_lmm_genetic_trend <- lmer(lmm_genetic_trend, data = df_data_filtered) # (So, n.d.).

# View the summary of the fitted LMM for the genetic trend.
summary(fit_lmm_genetic_trend) # (So, n.d.).

# Plot the residuals of the fitted LMM for the genetic trend.
plot(fit_lmm_genetic_trend, main = "Residual plot of the linear mixed model for the genetic trend", xlab = "Yield", ylab = "Residual Values") # (So, n.d.).

# Calculate and view the group adjusted yield means for "Cp."
df_group_adjusted_means_yfe <- emmeans(fit_lmm_genetic_trend, "Year_of_First_Entry") %>%
  broom::tidy() # (So, n.d.).
head(df_group_adjusted_means_yfe)

# Plot the group adjusted yield means for "Cp" against YFE to visualize the genetic trend (So, n.d.).
ggplot(data = df_group_adjusted_means_yfe, aes(x = Year_of_First_Entry, y = estimate, linetype = "genetic", colour = "genetic"), linewidth = 2) +
  geom_point(size = 3) +
  geom_smooth(data = df_group_adjusted_means_yfe, mapping = aes(x = Year_of_First_Entry, y = estimate), method = "lm", se = FALSE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90)) +
  labs(x = "Year of First Entry Into Trials", y = "Adjusted Yield Mean (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("P"), label.y = 0.9) +
  scale_x_continuous(breaks = seq(from = 1958, to = 2021, by = 1)) +
  scale_y_continuous(limits = c(2500, 5500)) +
  scale_linetype_manual(values = "solid", labels = "Genetic Trend", name = "") +
  scale_colour_manual(values = "#000099", labels = "Genetic Trend", name = "")
# The scatter plot shows that genetic factors slowly but steadily increase barley yields over the long term. Furthermore, the rate of yield increase as explained by genetic factors is estimated to be ~16.8 kg/ha/yr and is also significant, with p < 0.001.

# Create a new data frame containing the adjusted yield means from 1958 to 1989.
df_group_adjusted_means_yfe_1 <- df_group_adjusted_means_yfe %>% filter(Year_of_First_Entry <= 1989)

# Obtain the average adjusted yield mean from 1958 to 1989.
mean(df_group_adjusted_means_yfe_1$estimate)
# The average adjusted yield mean from 1958 to 1989 is 3482.516 kg/ha.

# Create a new data frame containing the adjusted yield means from 1990 to 2021.
df_group_adjusted_means_yfe_2 <- df_group_adjusted_means_yfe %>% filter(Year_of_First_Entry > 1989)

# Obtain the average adjusted yield mean from 1990 to 2021.
mean(df_group_adjusted_means_yfe_2$estimate)
# The average adjusted yield mean from 1990 to 2021 is 4019.255 kg/ha.

# ----- Obtaining the Genetic Trend by Variety -----

# Additionally, varieties can also be grouped based on the levels (i.e., years) of ri (Piepho et al., 2014). To do so, a categorical effect "Cp" can be defined as the adjusted variety-group means for groups "p = 1, 2, ..., P," where:
  # "P" is the number of levels of ri, and...
  # Each group "p" is comprised of all the varieties in a level of ri, which should include at least one variety (Piepho et al., 2014).

# As such, the genetic trend by variety can be modeled by "Gi = Cp + Hi," where:
  # Gi – The main effect of the ith variety,
  # Cp – A fixed categorical effect for groups "p = 1, 2, ..., P," where P is the number of levels of ri, and...
  # Hi – The random deviation of Gi from the genetic trend line, assumed to follow a normal distribution with a mean of zero and a variance of σ2H (Piepho et al., 2014).

# To do this, also treat "Variety" as a categorical effect (i.e., convert it into factors) (So, n.d.).

# Create another LMM to model the genetic trend by variety.
lmm_genetic_trend_per_variety <- formula(Yield ~ as.factor(Variety) + as.factor(Year_of_First_Entry) + (1|Location) + Year + (1|Location:Year) + (1|Variety:Location) + (1|Variety:Year)) # (So, n.d.).

# Fit the LMM for the adjusted yield means by variety to the filtered dataset.
fit_lmm_genetic_trend_per_variety <- lmer(lmm_genetic_trend_per_variety, data = df_data_filtered)

# Calculate and view the group adjusted yield means by variety.
emm_group_adjusted_means_yfe_per_variety <- emmeans(fit_lmm_genetic_trend_per_variety, specs = pairwise ~ Variety|Year_of_First_Entry, type = "response", rg.limit = 1000000) # (Muldoon, 2019).
df_group_adjusted_means_yfe_per_variety <- emm_group_adjusted_means_yfe_per_variety$emmeans %>%
  as.data.frame() # (Muldoon, 2019).
head(df_group_adjusted_means_yfe_per_variety)

# Plot the group adjusted entry yield means per year by variety against YFE to visualize the genetic trend by variety. The plot will be similar to Fig. 3 by So et al. (2022).
ggplot(data = df_group_adjusted_means_yfe_per_variety, aes(x = Year_of_First_Entry, y = emmean, linetype = "genetic", colour = "genetic"), linewidth = 2) +
  geom_point(size = 3) +
  geom_smooth(data = df_group_adjusted_means_yfe_per_variety, mapping = aes(x = Year_of_First_Entry, y = emmean), method = "lm", se = FALSE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90)) +
  labs(x = "Year of First Entry Into Trials", y = "Adjusted Yield Mean (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("P"), label.y = 0.9) +
  scale_x_continuous(breaks = seq(from = 1958, to = 2021, by = 1)) +
  scale_y_continuous(limits = c(2500, 5500)) +
  scale_linetype_manual(values = "solid", labels = "Genetic Trend", name = "") +
  scale_colour_manual(values = "#000099", labels = "Genetic Trend", name = "")
# The rate of yield increase by variety is estimated to be ~19.2 kg/ha/yr and is also significant, with p < 0.001.

# ----- Obtaining the Environmental Trend -----

# The model "Yk = γtk + Zk" can also be used to model the environmental trend, where:
  # Yk – The main effect of the kth year,
  # γ – A fixed regression coefficient for the environmental trend,
  # tk – A continuous covariate for the calendar year, and...
  # Zk – A random residual assumed to follow a normal distribution with a mean of zero and a variance of σ2Z (Piepho et al., 2014).

# As such, modeling the environmental trend requires the adjusted year means for Yk to be obtained using the model "yijk = μ + Gi + Lj + Yk + (LY)jk + (GL)ij + (GY)ik + (GLY)ijk" (Piepho et al., 2014).

# To do this, treat 1) "Variety" as a fixed effect to remove the effect of genetic factors and 2) "Year" as a fixed effect to actually obtain the adjusted year means (So, n.d.).

# Create another LMM to model the environmental trend.
lmm_environmental_trend <- formula(Yield ~ Variety + (1|Location) + Year + (1|Location:Year) + (1|Variety:Location) + (1|Variety:Year)) # (So, n.d.).

# Fit the LMM for the environmental trend to the filtered dataset.
fit_lmm_environmental_trend <- lmer(lmm_environmental_trend, data = df_data_filtered) # (So, n.d.).

# View the summary of the fitted LMM for the environmental trend.
summary(fit_lmm_environmental_trend) # (So, n.d.).

# Plot the residuals of the fitted LMM for the environmental trend.
plot(fit_lmm_environmental_trend, main = "Residual plot of the linear mixed model for the environmental trend", xlab = "Yield", ylab = "Residual Values") # (So, n.d.).

# Calculate and view the group adjusted yield means for years.
df_group_adjusted_means_year <- emmeans(fit_lmm_environmental_trend, "Year", rg.limit = 15000) %>%
  broom::tidy() %>%
  mutate(Year = as.numeric(Year)) # (So, n.d.).
head(df_group_adjusted_means_year)

# Plot the group adjusted yield means for years against calendar year to visualize the barley yield trend as explained by environmental factors (So, n.d.).
ggplot(data = df_group_adjusted_means_year, aes(x = Year, y = estimate, linetype = "environmental", colour = "environmental"), linewidth = 2) +
  geom_point(size = 3) +
  geom_smooth(data = df_group_adjusted_means_year, mapping = aes(x = Year, y = estimate), method = "lm", se = FALSE, linewidth = 1) +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90)) +
  labs(x = "Year of Trial", y = "Adjusted Yield Mean (kg/ha)") +
  stat_poly_eq(use_label("eq")) +
  stat_poly_eq(use_label("P"), label.y = 0.9) +
  scale_x_continuous(breaks = seq(from = 1958, to = 2021, by = 1)) +
  scale_y_continuous(limits = c(2500, 5500)) +
  scale_linetype_manual(values = "solid", labels = "Environmental Trend", name = "") +
  scale_colour_manual(values = "#006600", labels = "Environmental Trend", name = "")
# The scatter plot shows that environmental factors influence barley yields erratically year to year. Nevertheless, the rate of yield increase as explained by environmental factors is significant, with p < 0.001, and is estimated to be ~13.6 kg/ha/yr.

# Create a new data frame containing the adjusted yield means from 1958 to 1989.
df_group_adjusted_means_year_1 <- df_group_adjusted_means_year %>% filter(Year <= 1989)

# Obtain the average adjusted yield mean from 1958 to 1989.
mean(df_group_adjusted_means_year_1$estimate)
# The average adjusted yield mean from 1958 to 1989 is 3523.143 kg/ha.

# Create a new data frame containing the adjusted yield means from 1990 to 2021.
df_group_adjusted_means_year_2 <- df_group_adjusted_means_year %>% filter(Year > 1989)

# Obtain the average adjusted yield mean from 1990 to 2021.
mean(df_group_adjusted_means_year_2$estimate)
# The average adjusted yield mean from 1990 to 2021 is 4062.863 kg/ha.

# ----- References -----

# Laidig, F., Drobek, T., & Meyer, U. (2008). Genotypic and environmental variability of yield for cultivars from 30 different crops in German official variety trials. Plant Breeding, 127(6), 541-547. https://doi.org/10.1111/j.1439-0523.2008.01564.x
# Piepho, H., Laidig, F., Drobek, T., & Meyer, U. (2014). Dissecting genetic and non-genetic sources of long-term yield trend in German official variety trials. Theoretical and Applied Genetics, 127(5), 1009–1018. https://doi.org/10.1007/s00122-014-2275-1
# So, D. (n.d.). Chapter 3 - Trend Modeling [R script]. The R Foundation.
# So, D., Smith, A., Sparry, E., & Lukens, L. (2022). Genetics, not environment, contributed to winter wheat yield gains in Ontario, Canada. Theoretical and Applied Genetics, 135(6), 1893–1908. https://doi.org/10.1007/s00122-022-04082-3
