################### Assignment ARules on groceries dataset

install.packages('arules')
library(arules)
library(arulesViz) # for visualizing 

data('Groceries')
class(Groceries)
str(Groceries)

summary(Groceries)
# 9835 transactions have been done by customers
# the total products (or items) are 169
# top product purchased is whole milk. It has been purchased in 2513 transactions
# In 2519 transactions only one product has been purchased. In 3 transactions each
# 29 items has been purchased.
# minimum one item has been purchaded and maximum 32 items have been purchased
# per transaction.

inspect(Groceries[1:4]) # viewing first 4 transactions

# visualizing top20 transactions
itemFrequencyPlot(Groceries,topN=20)

# let us now build rules using apriori package
arules <- apriori(Groceries, parameter=list(support=0.002,confidence=0.6,minlen=2))
# support=0.002 means atleast 20 times in entire 10,000(9835) trans it should be present
arules # set of 376 rules 
inspect(head(sort(arules,by='lift')))
# get top 6 rules as per highest lift

head(quality(arules))
inspect(head(arules))

# visualize
plot(arules)
# when support decreases conf also decreases. 

# keep changing support and lift
arules1 <- apriori(Groceries,parameter=list(support=0.002,confidence=0.7,minlen=3))
# for minlen=3, get 94 rules
# for minlen=4, get 8 rules
# for minlen=5, get 5 rules
arules1a <- apriori(Groceries,parameter = list(support=0.002,confidence=0.75,minlen=3))
# for conf=0.7, get 94 rules
# for conf=0.75, get 39 rules
# for conf=0.8, get 11 rules
arules3 <- apriori(Groceries,parameter=list(support=0.003,confidence=0.7))
# 19 rules
arules3[is.redundant(arules3)] # 0 rules
inspect(arules3)

arules2 <- apriori(Groceries,parameter = list(support=0.002,confidence=0.8,minlen=3))
inspect(arules2)
inspect(sort(arules2,by='lift'))
plot(arules2)
# have 2 rules with high lift
plot(arules2,method = 'grouped')
plot(arules2,method = 'graph')

arules2_df = data.frame(
  lhs = labels(lhs(arules2)),
  rhs = labels(rhs(arules2)), 
  arules2@quality)

write.csv(arules2_df,file='E:\\Association_Rules\\arules2.csv')
# unlink('E:\\Association_Rules\\arules2a.csv')


# removing redundant rules
arules2[is.redundant(arules2)]
# set of 0 rules, there are no redundant rules

# arules2_redn <- arules2[!is.redundant(arules2)]
inspect(arules2_redn)

arules # set of 376 rules 
arules[is.redundant(arules)] # there are 6 redundant rules

# library(readxl)
library(writexl)
write_xlsx(arules2_df,'E:\\Association_Rules\\arules2.xlsx')
