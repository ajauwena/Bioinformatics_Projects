"""
==============================================================================================
Organize scripts
==============================================================================================
Authors:
    Abelhard Jauwena, ajauwena@gmail.com
    Olusegun Odumosu, odumosu.segun@gmail.com
    Nishita Sharif, nishita.sharif@gmail.com
Date: April 21, 2023
"""

import os



       
       
#-----------------------
# organize SVC files
#-------------------------
files = os.listdir() 
# create the ADAboost directory
os.mkdir('SVC')

# loop through each file
for file in files:
    # check if the file starts with "Adaboost"
    if file.startswith('SVC'):
        # move the file to the ADAboost directory
        os.rename(file, os.path.join('SVC', file))


#-----------------------
# organize SVR files
#-------------------------
files = os.listdir() 
# create the ADAboost directory
os.mkdir('SVR')

# loop through each file
for file in files:
    # check if the file starts with "Adaboost"
    if file.startswith('SVR'):
        # move the file to the ADAboost directory
        os.rename(file, os.path.join('SVR', file))



#-----------------------
# organize AdaBoost files
#-------------------------
files = os.listdir() 
# create the ADAboost directory
os.mkdir('AdaBoost')

# loop through each file
for file in files:
    # check if the file starts with "Adaboost"
    if file.startswith('AdaBoost'):
        # move the file to the ADAboost directory
        os.rename(file, os.path.join('AdaBoost', file))
        


#-----------------------
# organize KNN files
#-------------------------
files = os.listdir() 
# create the ADAboost directory
os.mkdir('KNN')

# loop through each file
for file in files:
    # check if the file starts with "Adaboost"
    if file.startswith('KNN'):
        # move the file to the ADAboost directory
        os.rename(file, os.path.join('KNN', file))