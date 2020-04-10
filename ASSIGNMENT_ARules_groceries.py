# ASSIGNMENT ASSOCIATION_RULES groceries dataset

# importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


groceries=[]
with open('E:/Datasets/groceries.csv') as f:
    groceries = f.read()
groceries=groceries.split('\n')    
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(','))

groceries_df = pd.DataFrame(pd.Series(groceries_list))
groceries_df.columns=['Transactions']
groceries_df[9833:]
groceries_df = groceries_df.iloc[:9835,:]
X = groceries_df.Transactions.str.join(sep='*').str.get_dummies(sep='*')

# getting a list of frequent items with atleast 3% frequency in the dataset
frequent_itemsets = apriori(X,  min_support=0.03,use_colnames=True)
frequent_itemsets.shape # 64 
frequent_itemsets.sort_values('support',ascending=False).head(5)
#Quick look at the top 25 items
X.sum().to_frame('Frequency').sort_values('Frequency',ascending=False)[:25].plot(kind='bar',
                                                                                 figsize=(12,8),title="Frequent Items");plt.show();

''' we have 64 items that appear atleast 3% of the times in the groceries
dataset. whole milk has highest support of 0.25 i.e., it appears 
25% times in the dataset, one-fourth of the transactions are made of
whole milk: followed by other-vegetables which appear 19% of the times.
'''

# Getting rules from the frequent itemsets
rules=association_rules(frequent_itemsets, metric='lift',min_threshold=1)
rules.shape # we have obtained 34 rules 
rules.head()

############# removing redundant rules
'''
we find that rules 0 and 1 are duplicate rules. similarly rule 2 and 3 are same
rules. we have to remove such redundant rules
'''
def to_list(i):
    return (sorted(list(i)))
ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redundancy  = rules.iloc[index_rules,:]

# Sorting them with respect to lift and getting top 10 rules 
rules_no_redundancy.sort_values('lift',ascending=False).head(10)
rules.shape # we have 34 rules including duplicate rules
rules_no_redundancy.shape # now we have 17 rules after removing duplicate rules
rules_no_redundancy.sort_values('lift',ascending=False)
'''
The top-most rule is root-vegetables and other-vegetables. this rule appears
4.7% of the times in the whole transactions. when a person purchases root-vegs,
we can be 43% confident that he will also purchase other-vegs. 
If Lift>1 indicates positive dependence or complementary effect. 
The lift valus is 2.25 indicating that they are positively correlated.

'''
# visualization
%matplotlib inline
# support vs confidence
plt.scatter(rules_no_redundancy.support, rules_no_redundancy.confidence, marker='*') 
plt.xlabel('support'); plt.ylabel('confidence'); plt.title('Support vs Confidence')

# support vs lift
plt.scatter(rules_no_redundancy.support, rules_no_redundancy.lift, marker='*') 
plt.xlabel('support'); plt.ylabel('lift'); plt.title('Support vs Lift')

# lift vs confidence
plt.scatter(rules_no_redundancy.lift, rules_no_redundancy.confidence)
# positive correlation is visible from the scatter plot
fit = np.polyfit(rules_no_redundancy['lift'], rules_no_redundancy['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules_no_redundancy['lift'], rules_no_redundancy['confidence'], 'yo', rules_no_redundancy['lift'], 
 fit_fn(rules_no_redundancy['lift']))
plt.xlabel('lift');plt.ylabel('confidence');plt.title('Lift vs Confidence');plt.show();

#### RECOMMENDATIONS
'''
1. those who buy root-vegs we can recommend to them other-vegs, whole-milk,fruits
2. Looking at all 17 rules, we see that root-vegs, other-vegs, whole-milk,
    rolls/buns, fruits(tropical and pip fruit),soda, bottled water, yogurt 
    have lift>1.19
    we can place these items in side-by-side racks.

'''

