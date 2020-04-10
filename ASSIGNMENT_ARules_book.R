########## ASSIGNMENT ASSOCIATION RULES on book dataset

library(arules)
library(arulesViz)

book <- read.csv("E:\\Association_Rules\\book.csv")
# there are 2000 transactions of 11 book-types
head(book,4)

rules <- apriori(as.matrix(book),parameter=list(support=0.002,confidence=0.7,minlen=3))
# get 3674 rules
rules <- apriori(as.matrix(book),parameter=list(support=0.05,confidence=0.8,minlen=3))
# get 61 rules when change supp=0.05 and conf=0.80

rules <- apriori(as.matrix(book),parameter=list(support=0.1,confidence=0.7,minlen=3))
inspect(rules)
arules[is.redundant(rules)] # no duplicate rules
plot(rules)
plot(rules,method='grouped')
rules_sort=sort(rules,by='lift')
inspect(rules_sort)
plot(rules_sort[1:2],method='graph')

######### Recommendations
# First Recommendation
# those who have taken DoItYBks with Art-books or Geog-books, we can refer them Cook-books
# since more than 10% of the times (support=0.1), the customers have bought these books.
# also we can be 80% confident, that they will go with these recommendations
# 
# second Recommendation
#those taken Art-books and Geog-books, we can recommend them either cook-books or child-books







