#Load in data
hiData <- read.csv("carine_spread.csv", header = TRUE)
hiData[] <- lapply(hiData, as.character)
rowNames <- hiData[1]

#Knock out data that is not absolute
hiData[hiData=="<10"] = "-1"
hiData[hiData=="<20"]<- "-1"  
hiData[hiData=="*"] <- "-1"

#Convert to numeric for operations
hiData[] <- lapply(hiData, as.numeric)
hiData[is.na(hiData)] <- -1

#Init new df
df <- data.frame(matrix(nrow = nrow(hiData)))

#Step 2 avaerage columns
#for (x in seq(2,ncol(hiData),2)) {
#  #print((hiData[x] + hiData[x+1])/2)
#  avgColumn <- (hiData[x] + hiData[x+1])/2
#  df <- cbind(df, avgColumn)
#}
#Step 2 average one by 1
for (x in seq(2,ncol(hiData),2)) {
  avgColumn <- vector()
  for (y in seq(1, nrow(hiData))) {
    #print((hiData[x] + hiData[x+1])/2)
    val1 = hiData[y, x]
    val2 = hiData[y,x+1]
    if (val1 == -1) { val1 = val2 }
    if (val2 == -1) { val2 = val1 }
    avgColumn[y] <- (val1 + val2)/2
  }
  df <- cbind(df, avgColumn)
  #Set column name
  colnames(df)[(x/2)+1] <- colnames(hiData)[x]
}

#Put row names back in
df[1] <- rowNames[1]

#Solve D(i,j) on column basis, where homologous titer is on the diagonal #THIS IS NOT TRUE
for (x in seq(2,ncol(df))) {
  #Find homo titer
  currentAntisera <- colnames(df)[x]
  currentAntisera <- gsub(".","/", currentAntisera, fixed = TRUE)
  homoIndex = which(df[1] == currentAntisera)
  
  #If multiple hits, use first
  if(length(homoIndex) > 1)
  {
    homoIndex <- homoIndex[1]
  }
  
  #Log2(Thomo) - Log2(Thetero)
  df[x] = log(df[homoIndex,x],2) - log(df[x],2) 
}

#Write averaged table out
write.csv(df, "carine_titers.csv")

