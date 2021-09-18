library(tidyverse)
library(multcomp)
df<-read.csv("try.csv")   # read in file
df$Group<-as.factor(df$Group) #treat groups as a factor, not a character

df$Group<- factor(df$Group, levels = c( "Control","Group_1","Group_2"))  ## Order the groups to compare against WT Diet 1 first
df %>% summarize(Group)  # check the order of the groups 

rq<-glm(RQ ~ Group, family = gaussian (link = "identity"), data=df)
print(summary(rq))

ee<-glm(EE ~ mass + Group + mass:Group, family = gaussian (link = "identity"), data=df)
print(summary(EE))

tukey <- glht(ee, mcp( Group = "Tukey" ))
print(summary(tukey))