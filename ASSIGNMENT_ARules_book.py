# ASSIGNMENT ASSOCIATION_RULES book dataset

# importing important libraries
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',500)
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

book = pd.read_csv("E:/Association_Rules/book.csv")
book.head()
book.shape # 2000,11
# There are 2000 transactions and 11 items


#Quick look at the top 10 items
%matplotlib qt
book.sum().to_frame('Frequency').sort_values('Frequency',ascending=False)[:10].plot(kind='bar',figsize=(12,8),title='Frequent Items')
'''cook-books are most popular followed by child-books and doltY-books
'''

# getting frequent items
frequent_items1 = apriori(book,  min_support=0.1,use_colnames=True)
frequent_items1.shape # 39 popular books
frequent_items1.sort_values('support',ascending=False).head(5) 

# Getting rules from the frequent itemsets
rules1=association_rules(frequent_items1, metric='lift',min_threshold=1)
rules1.shape # we have obtained 100 rules with 9 books(items)
rules1.sort_values('lift',ascending=False).head(5) 

# removing redundant rules
def sort_rules(i):
    return sorted(list(i))
rules_book = rules1.antecedents.apply(sort_rules) + rules1.consequents.apply(sort_rules)
rules_book = rules_book.apply(sorted)   
rules_sets = list(rules_book)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redundancy 
rules_no_redundancy  = rules1.iloc[index_rules,:]
rules1.shape # we have 100 rules including duplicate rules
rules_no_redundancy.shape # now we have 30 rules after removing duplicate rules

#### By changing the metric to confidence
rules2 = association_rules(frequent_items1,metric='confidence',min_threshold=0.8)
# the rules obtained by changing confidence to 0.6 is 30, 0.7 is 19, 0.8 is 6
# (these include duplicate rules)


df= rules_no_redundancy.sort_values('confidence',ascending=False).head(5)
df.to_excel('C:\\Users\\Admin\\Desktop\\df.xlsx',index=False)

''' please see the excel sheet for comparison
When we sort by SUPPORT, top rule is ChildBooks ==> Cookbooks
This has a support of 0.25. This only tells that both
books were purchased together for every 4th transaction (500 out of 2000). but when
we look at its confidence (0.6), indicates that those who bought ChildBooks
also bought CookBooks only 60% of the times.

When we sort by CONFIDENCE, top rule is ItalCook ==> CookBooks
the confidence is 1.0 indicating that whenever ItalCook was purchased,
Cookbooks were always purchased. But this does not include prior probability
of consequent. we see that support of only CookBooks is 0.431, indicating
that whatever is the antecedent, about 40% of the times CookBooks has been
purchased

When we take sort by  lift, top rule is ItalCook ==> CookBooks
when we take lift into consideration, it also gives weightage for
consequent support unlike confidence. here lift is 2.32 indicating that
both items are positively correlated.  
'''

#################### VISUALIZATION
# support vs confidence
plt.scatter(rules1['support'], rules1['confidence'], alpha=0.5,marker='*')
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show();

# support vs lift
plt.scatter(rules1['support'], rules1['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')
plt.show();

# lift vs confidence
plt.scatter(rules1['lift'], rules1['confidence'], alpha=0.5,marker='*')
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('Lift vs Confidence')
plt.show();


# Sorting them with respect to lift 
rules_no_redundancy = rules_no_redundancy.sort_values('lift',ascending=False)

######### RECOMMENDATIONS
'''
1. The topmost rule is ItalCook ==> CookBooks
    all who have purchased Italy-Cook_book have also purchased CookBooks. 
    So for customers who have purchased only Ital_Cook, we can recommend
    Cook-Books
2. The 2nd top rule is CookBooks,DoltYBooks ==> ArtBooks
    those who have purchased 2 of the 3 books, we can recommend 
    them to purchase 3rd book (since lift is 2.24, they are more likely to
    go with the recommendation)
3. We can bundle both italCook and CookBooks. Since this rule has confidence of 0.1,
    rather than recommending later, we can sell this combo at no extra-discount
4. we find that Florence has support of 0.10. it does not appear in the top rules
    Also, ArtBooks appear in many of the top rules. So we can sell both these
    as a combo with discount
'''











