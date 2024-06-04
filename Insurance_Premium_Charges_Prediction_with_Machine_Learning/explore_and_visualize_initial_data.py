"""
=========================================
Exploring and Visualizing the Raw Data
=========================================
Authors:
    Nishita Sharif, nishita.sharif@gmail.com
Date: April 21, 2023

How to run:   python3  explore_and_visualize_initial_data.py -in insurance_dataset.csv

Output figures:
Characteristics_of_Dataset.png
=========================================
"""
# Importing modules
import pandas as pd
import argparse
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec



##################
# set font sizes #
##################
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Defining command line arguments
parser = argparse.ArgumentParser(description='Multi-Layer Perceptron for classification and regression')
parser.add_argument('--in_file', '-in', action="store", dest ='in_file')
args = parser.parse_args()

filename      = args.in_file
insurance_premium_df = pd.read_csv(filename, header=0, encoding="utf8")

# Creating a figure that will contain 7 plots
fig = plt.figure(figsize=(10, 11)) # adjust the figure size as needed

gs=GridSpec(4,2) # 2 rows, 3 columns
gs.update(top=0.93, bottom=0.05, hspace=0.5) # adjust the top, bottom, and hspace

ax1=fig.add_subplot(gs[0,0]) # First row, first column
ax2=fig.add_subplot(gs[0,1]) # First row, second column
ax3=fig.add_subplot(gs[1,0]) # Second row, first column
ax4=fig.add_subplot(gs[1,1]) # Second row, second column
ax5=fig.add_subplot(gs[2,0]) # Third row, first column
ax6=fig.add_subplot(gs[2,1]) # Third row, second column
ax7=fig.add_subplot(gs[3,:]) # Fourth row, second column


# region Age
# Plotting a histogram of the age distribution of our data (using the values.tolist() function as the data input) 
ax1.hist(insurance_premium_df["age"].values.tolist(), 
         density=True, 
         bins = 15, 
         color ="#9dc6d8", 
         ec='black')
ax1.set_ylabel("Frequency")
ax1.set_xlabel("Age")
ax1.axvline(insurance_premium_df["age"].mean(), 
            color='k', 
            linestyle='dashed', 
            linewidth=1) # Adding line that represents mean
ax1.text(41, 0.03,f'Mean = {insurance_premium_df["age"].mean():.2f}',fontsize=10)
ax1.set_title("Age Distribution")

# Creating dataframe with summary statistics for age
age_summary_df = pd.DataFrame(insurance_premium_df.loc[:,"age"].describe())
# print(age_summary_df) #Can un-comment if want to print results
# endregion Age


# region Sex
# Plotting a pie chart of the sex distribution of our data 
sex_count_list= insurance_premium_df["sex"].value_counts().tolist()
sex_list = insurance_premium_df["sex"].value_counts().keys().str.capitalize().tolist()


sliceColors = ['#00b3ca','#7dd0b6']
ax2.pie(sex_count_list, 
        labels = sex_list, 
        colors = sliceColors, 
        autopct='%.2f%%', 
        startangle=90,
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 1,
                      'antialiased': True})
ax2.set_title("Sex Distribution")

# endregion Sex


# region num_children
# Plotting a histogram of the distribution number of children of individuals in our data 
frequency=insurance_premium_df["children"].value_counts().tolist()
num_children=insurance_premium_df["children"].value_counts().keys().tolist()

ax3.bar(num_children, 
        frequency,
         color ="#9dc6d8", 
         ec='black')
ax3.set_ylabel("Frequency")
ax3.set_xlabel("Number of children")
ax3.axvline(insurance_premium_df["children"].mean(), 
            color='k', 
            linestyle='dashed', 
            linewidth=1)
ax3.text(1.4, 400,f'Mean = {insurance_premium_df["children"].mean():.2f}',fontsize=10)
ax3.set_title("Distribution for Number of Children")

children_summary_df = pd.DataFrame(insurance_premium_df.loc[:,"children"].describe())
# print(children_summary_df) #Can un-comment if want to print results
# endregion num_children




# region Smoker
# Plotting a pie chart of the distribution of smoker vs nonsmoker in our data 
smoker_count_list = insurance_premium_df["smoker"].value_counts().tolist()
smoker_list = insurance_premium_df["smoker"].value_counts().keys().str.capitalize().tolist()

sliceColors = ['#e38690','#f69256']
ax4.pie(smoker_count_list, 
        labels = smoker_list, 
        colors = sliceColors, 
        startangle=90,
        autopct='%.2f%%', 
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 1,
                      'antialiased': True})
ax4.set_title("Smoker Distribution")

# endregion Smoker




# region BMI
# Plotting a histogram of the BMI distribution of our data
ax5.hist(insurance_premium_df["bmi"].values.tolist(), 
         density=True, 
         bins = 20, 
         color ="#9dc6d8", 
         ec='black')
ax5.set_ylabel("Frequency")
ax5.set_xlabel("BMI")
ax5.axvline(insurance_premium_df["bmi"].mean(), 
            color='k', 
            linestyle='dashed', 
            linewidth=1)
ax5.text(32, 0.06,f'Mean = {insurance_premium_df["bmi"].mean():.2f}',fontsize=10)
ax5.set_title("BMI Distribution")


# Creating dataframe with summary statistics for age
BMI_summary_df = pd.DataFrame(insurance_premium_df.loc[:,"bmi"].describe())
# print(BMI_summary_df) #Can un-comment if want to print results
# endregion BMI


# region Region 
# Plotting a pie chart of the regional distribution of our data 
region_count_list = insurance_premium_df["region"].value_counts().tolist() 
region_list = insurance_premium_df["region"].value_counts().keys().str.capitalize().tolist()

sliceColors = ['#00b3ca','#7dd0b6','#e38690','#f69256']
ax6.pie(region_count_list, 
        labels = region_list, 
        colors = sliceColors, 
        startangle=90,
        autopct='%.2f%%', 
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 1,
                      'antialiased': True})
ax6.set_title("Regional Distribution")

# endregion Region



# region Charges
# Plotting a histogram of the insurance premium charge distribution of our data 
ax7.hist(insurance_premium_df["charges"].values.tolist(), 
         density=True, 
         bins = 15, 
         color ="#9dc6d8", 
         ec='black')
ax7.set_ylabel("Frequency")
ax7.set_xlabel("Insurance Premium Charge")
ax7.axvline(insurance_premium_df["charges"].mean(), 
            color='k', 
            linestyle='dashed', 
            linewidth=1) # Adding line that represents mean
ax7.text(15000, 0.000045,f'Mean = {insurance_premium_df["charges"].mean():.2f}',fontsize=10)
ax7.set_title("Charge Distribution")

# Creating dataframe with summary statistics for age
charges_summary_df = pd.DataFrame(insurance_premium_df.loc[:,"charges"].describe())

# endregion Charges

fig.suptitle('Insurance Premium Data Distribution', fontsize=18, fontweight='bold', y = 0.98)
# fig.tight_layout()
fig.savefig("Characteristics_of_Dataset.png", dpi = 600)
fig.show()


# References
# How to make a pie chart (ice cream). Computer Programming & Problem Solving. http://cs111.wellesley.edu/labs/lab15/how-to-make-a-pie-chart
# https://stackoverflow.com/questions/31671999/different-size-subplots-in-matplotlib

