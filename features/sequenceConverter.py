#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:50:57 2019

@author: michael.zeller
"""

class sequenceConverter:
	def __init__(self):
		pass
		
	def translateToPolarity(self, pathInput, pathOutput = ""):
		"""Loads a fasta file. Any line without a def annotaiton of '>'
		will be translated to polarity marker
		
		args:
			pathInput (str): Fasta file to translate
			pathOutput (str): Location to write the output to. Blank 
				echos to console.
		"""
		#Open output file if specified
		if (pathOutput != ""):
			fileOutput = open(pathOutput, "w+")
		
		with open(pathInput, 'r') as fileInput:
			for line in fileInput:
				
				#Skip def lines
				if line[0] == ">":
					if (pathOutput != ""):
						fileOutput.write(line)
					else:
						print(line)
					continue
				
				#translate all else
				line = self.translateSequenceToPolarity(line)
			
				#Write either to file or to console
				if (pathOutput == ""):
					print(line)
				else:
					fileOutput.write(line)
		
		fileInput.close()
		if (pathOutput != ""):
			fileOutput.close()	
		
		
	def translateToHydrophobics(self, pathInput, pathOutput = ""):
		"""Loads a fasta file. Any line without a def annotaiton of '>'
		will be translated to polarity marker
		
		args:
			pathInput (str): Fasta file to translate
			pathOutput (str): Location to write the output to. Blank 
				echos to console.
		"""
		#Open output file if specified
		if (pathOutput != ""):
			fileOutput = open(pathOutput, "w+")
		
		with open(pathInput, 'r') as fileInput:
			for line in fileInput:
				
				#Skip def lines
				if line[0] == ">":
					if (pathOutput != ""):
						fileOutput.write(line)
					else:
						print(line)
					continue
				
				#translate all else
				line = self.translateSequenceToHydrophobicity(line)
			
				#Write either to file or to console
				if (pathOutput == ""):
					print(line)
				else:
					fileOutput.write(line)
		
		fileInput.close()
		if (pathOutput != ""):
			fileOutput.close()	
			
	def translateSequenceToPolarity(self, sequence):
		"""Translates the amino acid alphabet to polarity, as given by the 
		Geneious documentation. N = non-polar, P = polar, A = polar-acidic,
		B = polar-basic.
		http://assets.geneious.com/manual/8.0/GeneiousManualsu41.html
		
		args:
			sequences (str): Sequence(s) to be have alphabet converted on
			
		Returns:
			string: Translated sequences
		"""
		inputString =  "GAVLIFWMPSTCYNQDEKRH"
		outputString = "NNNNNNNNNPPPPPPAABBB"
		transDic = str.maketrans(inputString, outputString)
		return(sequence.upper().translate(transDic))
		
	def translateSequenceToHydrophobicity(self, sequence):
		"""Translates the amino acid alphabet to polarity, as given by the 
		Geneious documentation. N = non-polar, P = polar, A = polar-acidic,
		B = polar-basic.
		
		args:
			sequences (str): Sequence(s) to be have alphabet converted on
			
		Returns:
			string: Translated sequences
		"""
		inputString =  "FLIYWVMPCAGTSKQNHEDR"
		outputString = "HHHHHHHHMMMMMMLLLLLL"
		transDic = str.maketrans(inputString, outputString)
		return(sequence.upper().translate(transDic))
		
def main():
	seqConv = sequenceConverter()
	seqConv.translateToPolarity("data/translated.fasta","data/translated_polar.fasta")
	#seqConv.translateToPolarity("test_cases/civ_a.fasta","test_cases/civ_a_polar.fasta")
	seqConv.translateToPolarity("data/mergedH3Carine/translatedHA1.fasta","data/mergedH3Carine/translatedHA1_polar.fasta")
	#seqConv.translateToHydrophobics("data/translated.fasta","data/translated_phobics.fasta")
	#seqConv.translateToHydrophobics("test_cases/civ_a.fasta","test_cases/civ_a_phobics.fasta")
	
if __name__ == "__main__": main()