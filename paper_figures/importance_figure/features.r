library(ggplot2)
library(viridis)

df <- read.csv("importances.csv")

#Attempt to aggregate feature importances
#library(reshape2)
#mdata <- aggregate(formula = . ~ feature_pos, data = df, FUN = sum)
#Todo, alter size of circle based on # of features.... maybe rbind to prior df?
#mdata2 <- aggregate(formula = . ~ feature_pos, data = df, FUN = length)


#With dplyr
library(dplyr)
dCount <- df %>% 
  count(feature_pos) 

dImp <- df %>%
  group_by(feature_pos) %>%
  summarise(Imp=sum(importance)) %>%
  select(feature_pos,Imp)
merged <- merge(dCount, dImp, by = "feature_pos")

#Set identity row to 0, convert to factor, resort array, convert identity back to 1
identityRow <- df  %>%
  dplyr::filter(feature_pos %in% c("identity"))
merged <- merged[-c(which(merged$feature_pos == "identity")),]

#Convert to numeric and sort
merged$feature_pos <- as.numeric(as.character(merged$feature_pos))

#Bubbles
tiff("test.tiff", units="mm", width=178, height=133.5, res=300, compression='lzw')
ggplot(merged, aes(x=feature_pos, y=Imp, size=n, fill=Imp)) +
  geom_point(alpha=0.7, shape = 21) +
  geom_text(aes(label=ifelse(Imp>0.008, feature_pos,'')),hjust=0.5,vjust=-1, size=4) +
  # ggtitle("Cumulative GINI feature importance") +
  labs(x = "Amino acid position", y = "Importance") +
  scale_fill_viridis(option="magma") + 
  scale_size_continuous(range = c(2,10)) +
  scale_x_continuous(minor_breaks = seq(0, 350, 25), breaks = seq(0, 350, 50)) +
  theme_minimal() 


# insert ggplot code
dev.off()

#Backmap to protein
#install.packages("colourvalues")
library(colourvalues)
merged$col <- colour_values(merged$Imp, palette = "magma", include_alpha = FALSE)
write.csv(merged,"colors.csv")

#PRINT SELE and COLOR statements
for (row in 1:nrow(merged)) {
  print(paste("select resi", merged$feature_pos,";"))
}

