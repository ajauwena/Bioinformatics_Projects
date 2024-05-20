# ALIGNMENT STATISTICS GRAPHER

# Load the appropriate packages.
library("ggplot2")
library("dplyr")

# Set the working directory to the desired directory. Then, check if the working directory is indeed the desired directory.
setwd("C:/Users/ajauw/OneDrive/Documents/m_a/w/education/master's/mbinf/w23/binf_6110/projects/p2/r_code")
getwd()

# Read the .csv files from the directory as data frames.
df_burbot_bwa <- read.csv('df_burbot_statistics_bwa.csv')
df_cod_bwa <- read.csv('df_cod_statistics_bwa.csv')
df_burbot_bowtie2 <- read.csv('df_burbot_statistics_bowtie2.csv')
df_cod_bowtie2 <- read.csv('df_cod_statistics_bowtie2.csv')

# Plot bar graphs to visualize the alignment rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
# Using the burbot reference genome and bwa.
ggplot() +
  geom_bar(data = df_burbot_bwa, aes(x = File_Name, y = Alignment_Completeness), stat = "identity", fill = "red") +
  ylim(0, 1) +
  labs(x = "Sample_ID", y = "Alignment_Rate", title = "Fig. 1. The alignment rates of the 10 burbot reads when aligned to the burbot reference genome using bwa")

# Using the cod reference genome and bwa.
ggplot() +
  geom_bar(data = df_cod_bwa, aes(x = File_Name, y = Alignment_Completeness), stat = "identity", fill = "blue") +
  ylim(0, 1) +
  labs(x = "Sample_ID", y = "Alignment_Rate", title = "Fig. 2. The alignment rates of the 10 burbot reads when aligned to the cod reference genome using bwa")

# Using the burbot reference genome and bowtie2.
ggplot() +
  geom_bar(data = df_burbot_bowtie2, aes(x = File_Name, y = Alignment_Completeness), stat = "identity", fill = "green") +
  ylim(0, 1) +
  labs(x = "Sample_ID", y = "Alignment_Rate", title = "Fig. 3. The alignment rates of the 10 burbot reads when aligned to the burbot reference genome using bowtie2")

# Using the cod reference genome and bowtie2.
ggplot() +
  geom_bar(data = df_cod_bowtie2, aes(x = File_Name, y = Alignment_Completeness), stat = "identity", fill = "yellow") +
  ylim(0, 1) +
  labs(x = "Sample_ID", y = "Alignment_Rate", title = "Fig. 4. The alignment rates of the 10 burbot reads when aligned to the cod reference genome using bowtie2")

# Calculate the average alignment rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
average_alignment_rate_burbot_bwa <- mean(df_burbot_bwa$Alignment_Completeness)
average_alignment_rate_cod_bwa <- mean(df_cod_bwa$Alignment_Completeness)
average_alignment_rate_burbot_bowtie2 <- mean(df_burbot_bowtie2$Alignment_Completeness)
average_alignment_rate_cod_bowtie2 <- mean(df_cod_bowtie2$Alignment_Completeness)

# Create the columns for a new data frame, which will contain the average alignment rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
Alignment_Method_1 <- c("Burbot_BWA", "Cod_BWA", "Burbot_Bowtie2", "Cod_Bowtie2")
Average_Alignment_Rate <- c(average_alignment_rate_burbot_bwa, average_alignment_rate_cod_bwa, average_alignment_rate_burbot_bowtie2, average_alignment_rate_cod_bowtie2)

# Create a new data frame containing the average alignment rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
df_average_alignment_rates <- data.frame(Alignment_Method_1, Average_Alignment_Rate)

# Plot a bar graph to visualize the average alignment rates of the 10 burbot reads when aligned to the burbot and cod reference genoms using bwa and bowtie2.
ggplot() +
  geom_bar(data = df_average_alignment_rates, aes(x = Alignment_Method_1, y = Average_Alignment_Rate), stat = "identity", fill = "purple") +
  labs(x = "Alignment_Method", y = "Average_Alignment_Rate", title = "Fig. 5. The average alignment rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2")

# Conduct a t-test to see whether there is a significant difference between the mean alignment success between the two reference genomes as well as the two software.
# Comparing the burbot vs. cod reference genome when using bwa.
t.test(df_burbot_bwa[, 8], df_cod_bwa[, 8])

# Comparing the burbot vs. cod reference genome when using bowtie2.
t.test(df_burbot_bowtie2[, 8], df_cod_bowtie2[, 8])

# Comparing bwa vs. bowtie2 when using the burbot reference genome.
t.test(df_burbot_bwa[, 8], df_burbot_bowtie2[, 8])

# Comparing bwa vs. bowtie2 when using the cod reference genome.
t.test(df_cod_bwa[, 8], df_cod_bowtie2[, 8])

# Plot bar graphs to visualize the error rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
ggplot() +
  geom_bar(data = df_burbot_bwa, aes(x = File_Name, y = Error_Rate), stat = "identity", fill = "red") +
  ylim(0, 0.1) +
  labs(x = "Sample_ID", y = "Error_Rate", title = "Fig. 6. The error rates of the 10 burbot reads when aligned to the burbot reference genome using bwa")

ggplot() +
  geom_bar(data = df_cod_bwa, aes(x = File_Name, y = Error_Rate), stat = "identity", fill = "blue") +
  ylim(0, 0.1) +
  labs(x = "Sample_ID", y = "Error_Rate", title = "Fig. 7. The error rates of the 10 burbot reads when aligned to the cod reference genome using bwa")

ggplot() +
  geom_bar(data = df_burbot_bowtie2, aes(x = File_Name, y = Error_Rate), stat = "identity", fill = "green") +
  ylim(0, 0.1) +
  labs(x = "Sample_ID", y = "Error_Rate", title = "Fig. 8. The error rates of the 10 burbot reads when aligned to the burbot reference genome using bowtie2")

ggplot() +
  geom_bar(data = df_cod_bowtie2, aes(x = File_Name, y = Error_Rate), stat = "identity", fill = "yellow") +
  ylim(0, 0.1) +
  labs(x = "Sample_ID", y = "Error_Rate", title = "Fig. 9. The error rates of the 10 burbot reads when aligned to the cod reference genome using bowtie2")

# Calculate the average error rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
average_error_rate_burbot_bwa <- mean(df_burbot_bwa$Error_Rate)
average_error_rate_cod_bwa <- mean(df_cod_bwa$Error_Rate)
average_error_rate_burbot_bowtie2 <- mean(df_burbot_bowtie2$Error_Rate)
average_error_rate_cod_bowtie2 <- mean(df_cod_bowtie2$Error_Rate)

# Create the columns for a new data frame, which will contain the average error rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
Alignment_Method_2 <- c("Burbot_BWA", "Cod_BWA", "Burbot_Bowtie2", "Cod_Bowtie2")
Average_Error_Rate <- c(average_error_rate_burbot_bwa, average_error_rate_cod_bwa, average_error_rate_burbot_bowtie2, average_error_rate_cod_bowtie2)

# Create a new data frame containing the average error rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2.
df_average_error_rates <- data.frame(Alignment_Method_2, Average_Error_Rate)

# Plot a bar graph to visualize the average alignment rates of the 10 burbot reads when aligned to the burbot and cod reference genoms using bwa and bowtie2.
ggplot() +
  geom_bar(data = df_average_error_rates, aes(x = Alignment_Method_2, y = Average_Error_Rate), stat = "identity", fill = "purple") +
  labs(x = "Alignment_Method", y = "Average_Error_Rate", title = "Fig. 10. The average error rates of the 10 burbot reads when aligned to the burbot and cod reference genomes using bwa and bowtie2")
