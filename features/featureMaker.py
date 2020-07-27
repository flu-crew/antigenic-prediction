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

class featureMaker:
	def __init__(self, fileAlignment, fileAntigenicDistance):
		"""Creates features from alignments and antigenic distances
		
		Args:
			param1 (str): Aligned amino acid sequence, fasta format
			param2 (str): Raw HI data with columns and antisera, rows as antigens
			
		"""
		self.__fileAlignment = fileAlignment
		self.__fileAntigenicDistance = fileAntigenicDistance
		self.featureList = None
		self.glycosylationList = None
		self.features = None
		
		#Load alignments
		self.__loadDistances()
		self.__loadAlignment()
		
	def gatherAminoAcidDifferences(self):
		"""Finds all pairwise amino acid differences in a dataset. Updates the featureList variable"""
		#Init feature list
		featureList = []
		glycList = []
		
		#Iterate through square antigenic distance matrix, grab features
		colNames = self.distances.columns.tolist()
		combos = list(itertools.combinations(colNames,2))
		for i in combos:
			antigenSeq = self.sequences.loc[i[0]][0]
			antiserumSeq = self.sequences.loc[i[1]][0]
			featureList = list(set(featureList + self.seqDiff(antigenSeq, antiserumSeq)))
			
		if self.featureList is None:
			self.featureList = featureList
		else:
			self.featureList = list(set(featureList + self.featureList))
				
	def gatherGlycosylationDifferences(self):
		"""Finds potential N-linked glycosylation sites across all sequences using N-X-S/T-X motif. Updates featureList variable."""
		#Init feature list
		glycList = []
		
		#Iterate through square antigenic distance matrix, grab glycosylation
		colNames = self.distances.columns.tolist()
		for i in colNames:
			antigenSeq = self.sequences.loc[i][0]
			for j in colNames:
				antiserumSeq = self.sequences.loc[j][0]
				glycList = list(set(glycList + self.findNGlycSites(antigenSeq, antiserumSeq)))
		
		if self.featureList is None:
			self.featureList = glycList
		else:		
			self.featureList = list(set(glycList + self.featureList))
	
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
				
				#Check feature type
				if(feature[0] == "g"):
					featureArray[iterator, k] = self.checkNGlycDiff(antigenSeq, antiserumSeq, feature[1:])
				else:
					#Check each feature using negative indeces
					position = int(feature[:-2]) - 1	#Off by one, index starts at 0
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

	def exportFeatures(self, outputFile):
		"""Writes features out to file in csv format."""
		self.features.to_csv(outputFile)
	
	def exportTestSetFeatures(self, outputFile):
		"""Writes features from test set out to csv format."""
		self.testSetFeatures.to_csv(outputFile)
		
	def defineFeatures(self):
		"""Tests the featureList across all pairwise sequence comparisons. Updates features variable for exporting."""
		#Create a data structure to fill in
		colNames = self.distances.columns.tolist()
		combos = list(itertools.combinations_with_replacement(colNames,2))
		arrLength = len(combos)
		arrWidth = len(self.featureList)
		nameArray = np.zeros((arrLength,2), dtype = "U64")
		featureArray = np.zeros((arrLength, arrWidth), dtype = np.int8)
		distanceArray = np.zeros((arrLength, 2), dtype = np.float)
		
		#Iterate through square antigenic distance matrix, grab features
		#combos = list(itertools.combinations(colNames,2))

		iterator = 0
		for i in combos:
			antigenSeq = self.sequences.loc[i[0]][0]
			antiserumSeq = self.sequences.loc[i[1]][0]
			
			#Skip if antigen == antiserum
			#if(i[0] == i[1]):
			#	continue
			
			#Fill out name table 
			nameArray[iterator, 0] = i[0]
			nameArray[iterator, 1] = i[1]
			
			#Fill out distance table
			distanceArray[iterator, 0] =  self.distances.loc[i[1],i[0]]
			distanceArray[iterator, 1] = self.seqIdentity(antigenSeq, antiserumSeq)
				
			#Check for features
			for k, feature in enumerate(self.featureList):
				
				#Check feature type
				if(feature[0] == "g"):
					featureArray[iterator, k] = self.checkNGlycDiff(antigenSeq, antiserumSeq, feature[1:])
				else:
					#Check each feature using negative indeces
					position = int(feature[:-2]) - 1	#Off by one, index starts at 0
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
		
		#Set internal features array
		self.features = df
	
	def __loadDistances(self):
		"""Loads distance matrix, assuming symmetric, into global variable"""
		self.distances = pd.read_csv(self.__fileAntigenicDistance, header=0, index_col=[0])
			
	def __loadAlignment(self):
		"""Loads alignment file and sets global variable"""
		#Add single pairwise  AA changes as features
		with open(self.__fileAlignment, 'r') as alignmentFile:
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
	
	def seqDiff(self, seq1, seq2):
		"""Find amino acid difference between two sequences. 
	
		Args:
			param1 (str): genetic sequence 1
			param2 (str): genetic sequence 2
			
		Returns: 
			list: position number and change.
			
		Todo: 
			Raises, need to throw errors if lengths are not the same
		"""
		seq1 = seq1.lower()
		seq2 = seq2.lower()
		columnList = []
		for j in range(0,len(seq1)):	#Assuming lengths are the same, because alignment
			if(seq1[j] != seq2[j]):
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
		
		Consider: Adding weighting based on BLOSUM62
		"""
		count = sum(1 for a, b in zip(seq1, seq2) if a != b)
		return (100 - (round(100*((len(seq1) - count)/len(seq1)),3)))
	

	def findNGlycSites(self, seq1, seq2):
		"""Find potential N-linked glycosylation sites in two sequences and return unshared glycosylation
		
		Args:
			param1 (str): genetic sequence 1
			param2 (str): genetic sequence 2
			
		Returns: 
			list: unshared glycosylation sites between two viruses.
		"""
		glycList = []
		pattern = r"n[^p][st][^p]"
		matches1 = [match.start() for match in re.finditer(pattern, seq1)]
		matches2 = [match.start() for match in re.finditer(pattern, seq2)]
		
		#Find differences between two lists
		glycList = list(set(matches1) - set(matches2))
		
		return glycList
		
	def checkNGlycDiff(self, seq1, seq2, location):
		"""Find N-linked glycosylation in two sequences and return unshared glycosylation
	
		Args:
			param1 (str): genetic sequence 1
			param2 (str): genetic sequence 2
			
		Returns: 
			bool: True if the number of potential glycosylation points has changed
		"""
		location = int(location)
		pattern = r"n[^p][st][^p]"
		matches1 = [match.start() for match in re.finditer(pattern, seq1[location - 1:location + 3])]
		matches2 = [match.start() for match in re.finditer(pattern, seq2[location - 1:location + 3])]
	
		#If number of hits vary, return true	
		if (len(matches1) != len(matches2)):
			return 1
		
		#Fall through to false
		return 0
		
	"""
	def betaCreateSquaredFeatures(self): #Memory error
		import itertools
		
		combos = list(itertools.combinations(h3Set.featureList,2))
		featureArray = np.zeros((len(h3Set.features.index), len(combos)), dtype = np.int8)
		nameArray = []	
		for key, item in enumerate(combos):
			featureArray[:,key] = h3Set.features[item[0]] * h3Set.features[item[1]]
			nameArray.append(item[0] + "_" +  item[1])
		
		#Stitch arrays together into a dataframe
		df = pd.DataFrame(np.zeros((len(h3Set.features.index), len(combos) + 4)),columns=['antigen','antiserum','dist','identity'] + nameArray)
		df["antigen"] = h3Set.features["antigen"].to_numpy()
		df["antiserum"] = h3Set.features["antiserum"].to_numpy()
		df["dist"] = h3Set.features["dist"].to_numpy()
		df["identity"] = h3Set.features["identity"].to_numpy()
		df.iloc[:,4:] = featureArray[:,:]
	"""