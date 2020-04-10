##################### ASSIGNMENT ASSOCIATION RULES on MOVIES dataset

library(arules)
library(arulesViz)

movies_raw <- read.csv('E:\\Association_Rules\\my_movies.csv')
head(movies_raw)
movies <- as.matrix(movies_raw[,-c(1,2,3,4,5)])
head(movies)
class(movies)

rules_movie <- apriori(movies,parameter=list(support=0.2,confidence=0.4))
# sup 0.22, conf 0.4 get 15 rules
# when do not give min length then get some rules with only consequent (no antecedent)
inspect(rules_movie)

rules_movie <- apriori(movies,parameter=list(support=0.1,confidence=0.4,minlen=2))
# sup 0.1, conf 0.4, minlen 2 , get 105 rules
# sup 0.1 indicates that movie should be present in 10% of transactions. since there
# are only 10 transactions, this implies that even if movie is present in only
# 1 transaction, the movie will be used to generate rules. This is not proper
inspect(rules_movie)

# remove redundant rules
'''
the first rule is  Harry.Potter2 => Harry.Potter1
the second rule is Harry.Potter1 => Harry.Potter2
both are same rules, we need to remove one duplicate rule
'''
rules_movie[is.redundant(rules_movie)] # there are 58 redundant rules
rules_movie_redund <- rules_movie[!is.redundant(rules_movie)] # removing redundant rules
rules_movie_redund # get 47 rules after removing 58 duplicate rules

rules_movie1 <- apriori(movies,parameter=list(support=0.2,confidence=0.4))
# sup 0.2, conf 0.4 get 15 rules
# when do not give min length then get some rules with only consequent (no antecedent)
inspect(rules_movie1)

rules_movie2 <- apriori(movies,parameter=list(support=0.2,confidence=0.4,minlen=2))
# sup 0.2, conf 0.4 get 108 rules
inspect(rules_movie2)

rules_movie2_df = data.frame(
  lhs = labels(lhs(rules_movie2)),
  rhs = labels(rhs(rules_movie2)), 
  rules_movie2@quality)

library(writexl)
write_xlsx(rules_movie2_df,'E:\\Association_Rules\\rules_movies.xlsx')
