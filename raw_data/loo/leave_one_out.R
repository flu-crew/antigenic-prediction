library(ggplot2)
library(reshape2)

df <- read.csv("dist_summary.tsv", sep = "\t")
mdf <- melt(df)

#Tranpose important data
tiff('test.tiff', units="mm", width=178, height=133.5, res=300, compression = 'lzw')

ggplot(data = mdf, aes(x = variable, y = value, fill = variable)) +
  geom_violin() +
  geom_boxplot(width=0.1) +
  xlab("Regressor") +
  ylab("Dist from expected (AU)") + 
  scale_y_continuous(breaks=seq(0, 10, 1)) + # Ticks from 0-10, every .25 
  theme(legend.position = "none", 
        panel.background = element_rect(fill = "white"),
        panel.grid.major = element_line(colour = "gray90"),
        panel.grid.minor = element_line(colour = "gray90"),
        axis.title = element_text(size=12,face="bold"),
        axis.text = element_text(size=11,face="bold"),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust=0.5)
  )
dev.off()

ggplot(data = mdf, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot() +
  theme(legend.position = "bottom", 
        panel.background = element_rect(fill = "white"),
        panel.grid.major = element_line(colour = "gray90"),
        panel.grid.minor = element_line(colour = "gray90"),
        axis.title = element_text(size=12,face="bold"),
        axis.text = element_text(size=11,face="bold"),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust=0.5)
  )
