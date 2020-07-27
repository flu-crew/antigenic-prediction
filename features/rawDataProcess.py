# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 13:33:00 2019

@author: michael.zeller
"""

import sys
import os
import getopt
import re
import csv
import time
import pandas as pd
import numpy as np
from io import StringIO
import itertools

class rawDataProcess:

	def __init__(self):
		"""Init features"""
		self.featureList = None
		self.features = None
		
	def loadFoldChangeTable(self, fileAlignment, fileFoldChange):
		"""Loads fold change matrix and process features"""
		self.distances = pd.read_csv(fileFoldChange, header=0, index_col=[0])
		
		#Add single pairwise  AA changes as features
		with open(fileAlignment, 'r') as alignmentFile:
			alignment = alignmentFile.read()
		alignment = alignment.replace("\r","")
		alignment = alignment.replace("\n",",")	#Check if there is a trailing ',' and remove? Else error on stringIO
		alignment = alignment.replace(",>","\n")
		alignment = alignment.replace(">","")
		alignment = "name,sequence\n" + alignment 	#Prepend headers
		alignment = alignment.lower()
		
		#Push into a dataframe
		sequenceList = StringIO(alignment)
		self.sequences = pd.read_csv(sequenceList, sep = ",", index_col=[0])

		#Init feature list, finds all pairwise amino acid differences in a dataset
		featureList = []

		#Iterate through square antigenic distance matrix, grab features
		colNames = self.distances.columns.tolist()
		rowNames = self.distances.index.tolist()
		combos = list(itertools.product(colNames,rowNames))

		#Check which sequences are missing
		errorFlag = 0
		seqNames = self.sequences.index.tolist() 
		for i in combos:
			if(not i[0] in seqNames):
				print("Missing Sequence {}", i[0])
				errorFlag = 1
			if(not i[1] in seqNames):
				print("Missing Sequence {}", i[1])
				errorFlag = 1
		if(errorFlag == 1):
			return
			
		#Grab features
		for i in combos:
			antigenSeq = self.sequences.loc[i[0]][0]
			antiserumSeq = self.sequences.loc[i[1]][0]
			featureList = list(set(featureList + self.seqDiff(antigenSeq, antiserumSeq)))
			
		if self.featureList is None:
			self.featureList = featureList
		else:
			self.featureList = list(set(featureList + self.featureList))
			
	def defineFeatures(self):
		"""Tests the featureList across all pairwise sequence comparisons. Updates features variable for exporting."""
		#Create a data structure to fill in
		colNames = self.distances.columns.tolist()
		rowNames = self.distances.index.tolist()
		combos = list(itertools.product(colNames,rowNames))
		arrLength = len(combos)
		arrWidth = len(self.featureList)
		nameArray = np.zeros((arrLength,2), dtype = "U64")
		featureArray = np.zeros((arrLength, arrWidth), dtype = np.int8)
		distanceArray = np.zeros((arrLength, 2), dtype = np.float)
		
		#Iterate through matrix, grab features
		iterator = 0
		for i in combos:
			antigenSeq = self.sequences.loc[i[0]][0]
			antiserumSeq = self.sequences.loc[i[1]][0]
					
			#Fill out name table 
			nameArray[iterator, 0] = i[0]
			nameArray[iterator, 1] = i[1]

			#Fill out distance table
			distanceArray[iterator, 0] = self.distances.loc[i[1],i[0]]
			distanceArray[iterator, 1] = self.seqIdentity(antigenSeq, antiserumSeq)
				
			#Check for features
			for k, feature in enumerate(self.featureList):
				
				#Check each feature using negative indeces
				position = int(feature[:-2]) - 1	#Off by one, index starts at 
				if(position > len(antigenSeq) or position > len(antiserumSeq)):		#Dodge string index out of range error
					continue
				if(antigenSeq [position] == feature[-1] and antiserumSeq[position] == feature[-2]):
					featureArray[iterator, k] = 1
				if(antigenSeq [position] == feature[-2] and antiserumSeq[position] == feature[-1]):
					featureArray[iterator, k] = 1

			#Increment Iterator
			iterator += 1
	
		#Stitch arrays together into a dataframe
		df = pd.DataFrame(np.zeros((arrLength, arrWidth + 4)),columns=['antigen','antiserum','dist','identity'] + self.featureList)
		#df.iloc[:,0] = nameArray[:,0]
		df.iloc[:,0:2] = nameArray[:,0:2]
		df.iloc[:,2:4] = distanceArray[:,0:2]
		df.iloc[:,4:] = featureArray[:,:]
		
		#Drop out data with fold change of -1
		df = df[df["dist"] != -1]
		
		#Set internal features array
		self.features = df

	def exportFeatures(self, outputFile):
		"""Writes features out to file in csv format."""
		self.features.to_csv(outputFile)
	
	def exportTestSetFeatures(self, outputFile):
		"""Writes features from test set out to csv format."""
		self.testSetFeatures.to_csv(outputFile)

	def createTestSet(self, inputAlignment, selfCompare = False):
		#Load test sequences
		with open(inputAlignment, 'r') as alignmentFile:
			alignment = alignmentFile.read()
		alignment = alignment.replace("\r","")
		alignment = alignment.replace("\n",",")	#Check if there is a trailing ',' and remove? Else error on stringIO
		alignment = alignment.replace(",>","\n")
		alignment = alignment.replace(">","")
		alignment = "name,sequence\n" + alignment 	#Prepend headers
		alignment = alignment.lower()
		
		#Push into a dataframe
		sequenceList = StringIO(alignment)
		testSequences = pd.read_csv(sequenceList, sep = ",", index_col=[0])
		
		#Create test set feature list 
		colNames = self.distances.columns.tolist()
		testCols = list(testSequences.index.values)
		#Merge test set to the training set if self comparison is desired
		if(selfCompare == True):
			self.sequences = pd.concat([self.sequences, testSequences])
			colNames = colNames + testCols
		combos = list(itertools.product(testCols,colNames))
		arrLength = len(combos)
		if(selfCompare == True):
			arrLength -= len(testCols)
		arrWidth = len(self.featureList)
		nameArray = np.zeros((arrLength,2), dtype = "U64")
		featureArray = np.zeros((arrLength, arrWidth), dtype = np.int8)
		distanceArray = np.zeros((arrLength, 2), dtype = np.float)

		#Iterate through square antigenic distance matrix, grab features
		iterator = 0
		for i in combos:
			antigenSeq = testSequences.loc[i[0]][0]
			antiserumSeq = self.sequences.loc[i[1]][0]
			
			#Skip if antigen == antiserum
			if(i[0] == i[1]):
				continue
			
			#Fill out name table 
			nameArray[iterator, 0] = i[0]
			nameArray[iterator, 1] = i[1]
			
			#Fill out distance table
			distanceArray[iterator, 0] =  0
			distanceArray[iterator, 1] = self.seqIdentity(antigenSeq, antiserumSeq)
				
			#Check for features
			for k, feature in enumerate(self.featureList):
				
				#Check each feature using negative indeces
				position = int(feature[:-2]) - 1	#Off by one, index starts at 0
				#if(position > len(antigenSeq) or position > len(antiserumSeq)):		#Dodge string index out of range error; skip wrong length strings
				#	continue
				if(antigenSeq [position] == feature[-1] and antiserumSeq[position] == feature[-2]):
					featureArray[iterator, k] = 1
				if(antigenSeq [position] == feature[-2] and antiserumSeq[position] == feature[-1]):
					featureArray[iterator, k] = 1

			#Increment Iterator
			iterator += 1
			print(iterator)

		#Stitch arrays together into a dataframe
		df = pd.DataFrame(np.zeros((arrLength, arrWidth + 4)),columns=['antigen','antiserum','dist','identity'] + self.featureList)
		#df.iloc[:,0] = nameArray[:,0]
		df.iloc[:,0:2] = nameArray[:,0:2]
		df.iloc[:,2:4] = distanceArray[:,0:2]
		df.iloc[:,4:] = featureArray[:,:]
		
		self.testSetFeatures = df


	def createUnknownComparison(self, inputAlignment):
		"""Used to build unknown-unknown comparison rather then comparison to the antisera set"""
		#Load test sequences
		with open(inputAlignment, 'r') as alignmentFile:
			alignment = alignmentFile.read()
		alignment = alignment.replace("\r","")
		alignment = alignment.replace("\n",",")	#Check if there is a trailing ',' and remove? Else error on stringIO
		alignment = alignment.replace(",>","\n")
		alignment = alignment.replace(">","")
		alignment = "name,sequence\n" + alignment 	#Prepend headers
		alignment = alignment.lower()
		
		#Push into a dataframe
		sequenceList = StringIO(alignment)
		testSequences = pd.read_csv(sequenceList, sep = ",", index_col=[0])
		
		#Create test set feature list 
		colNames = list(testSequences.index.values)
		testCols = list(testSequences.index.values)
		combos = list(itertools.product(testCols,colNames))
		arrLength = len(combos)
		arrWidth = len(self.featureList)
		nameArray = np.zeros((arrLength,2), dtype = "U64")
		featureArray = np.zeros((arrLength, arrWidth), dtype = np.int8)
		distanceArray = np.zeros((arrLength, 2), dtype = np.float)

		#Iterate through square antigenic distance matrix, grab features
		iterator = 0
		for i in combos:
			antigenSeq = testSequences.loc[i[0]][0]
			antiserumSeq = testSequences.loc[i[1]][0]
			
			#Skip if antigen == antiserum
			#if(i[0] == i[1]):
			#	continue
			
			#Fill out name table 
			nameArray[iterator, 0] = i[0]
			nameArray[iterator, 1] = i[1]
			
			#Fill out distance table
			distanceArray[iterator, 0] =  0
			distanceArray[iterator, 1] = self.seqIdentity(antigenSeq, antiserumSeq)
				
			#Check for features
			for k, feature in enumerate(self.featureList):
				
				#Check each feature using negative indeces
				position = int(feature[:-2]) - 1	#Off by one, index starts at 0
				if(position > len(antigenSeq) or position > len(antiserumSeq)):		#Dodge string index out of range error; skipe wrong length strings
					continue
				if(antigenSeq [position] == feature[-1] and antiserumSeq[position] == feature[-2]):
					featureArray[iterator, k] = 1
				if(antigenSeq [position] == feature[-2] and antiserumSeq[position] == feature[-1]):
					featureArray[iterator, k] = 1

			#Increment Iterator
			iterator += 1
			print(iterator)

		#Stitch arrays together into a dataframe
		df = pd.DataFrame(np.zeros((arrLength, arrWidth + 4)),columns=['antigen','antiserum','dist','identity'] + self.featureList)
		#df.iloc[:,0] = nameArray[:,0]
		df.iloc[:,0:2] = nameArray[:,0:2]
		df.iloc[:,2:4] = distanceArray[:,0:2]
		df.iloc[:,4:] = featureArray[:,:]
		
		self.testSetFeatures = df

	def seqDiff(self, seq1, seq2):
		"""Find amino acid difference between two sequences. 
	
		Args:
			param1 (str): genetic sequence 1
			param2 (str): genetic sequence 2
			
		Returns: 
			list: position number and change.
			
		"""
		#Pick smaller length for looping
		loopSize = len(seq1)
		if(len(seq2) < loopSize):
			loopSize = len(seq2)
			
		seq1 = seq1.lower()
		seq2 = seq2.lower()
		columnList = []
		for j in range(0,loopSize):	#Assuming lengths are the same, because alignment
			if(seq1[j] != seq2[j]):
				#If x, skip
				if(seq1[j] =="x" or seq2[j] == "x"):
					continue
				
				#Check if column exists, else create it
				feature = str(j + 1) + ''.join(sorted(seq1[j] + seq2[j]))	#Off by 1, index starts at 0
				
				columnList.append(feature)
							
		#Return modified dataframe
		return columnList

	def seqIdentity(self, seq1, seq2):
		"""Calculate raw identity between two sequences
		
		Args:
			param1 (str): genetic sequence 1
			param2 (str): genetic sequence 2
			
		Returns: 
			float: Percent similarity between two sequences
			
		Reference: https://stackoverflow.com/questions/28423448/counting-differences-between-two-strings
		
		Consider: Adding weightign based on BLOSUM62
		"""
		count = sum(1 for a, b in zip(seq1, seq2) if a != b)
		return (100 - (round(100*((len(seq1) - count)/len(seq1)),3)))

"""
#Test the code
dataProc = rawDataProcess()
dataProc.loadFoldChangeTable("raw_data/master.fasta", "raw_data/marcus_final.csv")
#dataProc.loadFoldChangeTable("raw_data\master.fasta", "raw_data\marcus_final.csv") #Try to load a 2nd table in so additional features can be had
#Remove symmetry assumptions
dataProc.defineFeatures()
dataProc.exportFeatures("raw_data/marcus_feature_set.csv")
dataProc.createTestSet("test_cases/screen/testset1.fasta", selfCompare = False, selfCompare = False)
dataProc.exportTestSetFeatures("raw_data/testset1.csv")

dp = rawDataProcess()
#dp.loadFoldChangeTable("raw_data/master.fasta", "raw_data/marcus_final.csv")
dp.loadFoldChangeTable("raw_data/master.fasta", "raw_data/carine_final.csv")
dp.defineFeatures()
dp.exportFeatures("raw_data/carine_feature_set.csv")
#dp.createTestSet("raw_data/testset2.fasta", selfCompare = False)
dp.createUnknownComparison("raw_data/testset1.fasta")
dp.exportTestSetFeatures("raw_data/crossset1.csv")
"""