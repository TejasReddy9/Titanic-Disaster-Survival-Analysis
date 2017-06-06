import numpy as np
import pandas as pd 
# from matplotlib import style
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import seaborn as sns

# style.use('ggplot')
# sns.set(style = 'white')
# pylab.rcParams['figure.figsize'] = 8, 6

# Helper Functions for Visualization
def plot_correlation_map(df):
	corr = df.corr()
	f, ax = plt.subplots(figsize = (12,10))
	cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
	_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = { 'fontsize' : 12 })
	plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    plt.show()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
    plt.show()


train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

titanic = train.append(test, ignore_index = True)  # Make full data
del train, test

# Get the data to work on.. containing 891 elements
titanic = titanic[:891]

# print titanic.head()
# print titanic.describe()

# ######## Visualizations
# # print titanic.head()
# # print titanic.describe()

# # To check which varaibles are important
# print titanic.corr() # We can also plot it as a correlation heat map

# plot_correlation_map(titanic)

# plot_distribution(titanic, var='Age', target='Survived', row='Sex')
# plot_distribution(titanic, var='Fare', target='Survived', row='Sex')
# plot_categories(titanic, cat='Embarked', target='Survived')
# plot_categories(titanic, cat='Sex', target='Survived')
# plot_categories(titanic, cat='Pclass', target='Survived')
# plot_categories(titanic, cat='SibSp', target='Survived')
# plot_categories(titanic, cat='Parch', target='Survived')
# ##########


# ########## Creating labels '1's and '0's
# title = pd.DataFrame(titanic.Name)
# # we extract the title from each name
# title['Title'] = title['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
# Title_Dictionary = {
#                     "Capt"			:	"Officer",
#                     "Col"			:	"Officer",
#                     "Major"			:	"Officer",
#                     "Jonkheer"		:   "Royalty",
#                     "Don"			:	"Royalty",
#                     "Sir" 			:	"Royalty",
#                     "Dr"			:	"Officer",
#                     "Rev"			:	"Officer",
#                     "the Countess"	:	"Royalty",
#                     "Dona"			:	"Royalty",
#                     "Mme"			:	"Mrs",
#                     "Mlle"			:	"Miss",
#                     "Ms"			:	"Mrs",
#                     "Mr" 			:	"Mr",
#                     "Mrs" 			:	"Mrs",
#                     "Miss" 			:	"Miss",
#                     "Master" 		:	"Master",
#                     "Lady" 			:	"Royalty"
#                 }
# # we map each title
# title['Title'] = title.Title.map(Title_Dictionary)
# title = pd.get_dummies(title.Title)
# print title.head()

# sex_series = pd.Series(np.where(titanic.Sex=='male', 1, 0), name='Sex')
# sex = pd.Series.to_frame(sex_series)
# print sex.head()

# embarked = pd.get_dummies(titanic.Embarked, prefix='Embarked')
# print embarked.head()

# Pclass = pd.get_dummies(titanic.Pclass, prefix='Pclass')
# print Pclass.head()
# # embarked = pd.get_dummies(titanic.embarked, prefix='Embarked')
# # embarked = pd.get_dummies(titanic.embarked, prefix='Embarked')
# # embarked = pd.get_dummies(titanic.embarked, prefix='Embarked')
# ##########


# titanic.Age.fillna(titanic.Age.mean())
# titanic.Fare.fillna(titanic.Fare.mean())
# print titanic.head()

# titanic.Cabin.fillna('U',inplace=True)
# cabin = pd.DataFrame(titanic.Cabin)
# cabin['Cabin'] = cabin['Cabin'].map(lambda c : c[0])
# cabin = pd.get_dummies(cabin.Cabin, prefix='Cabin')
# print cabin.head()

def cleanTicket(ticket):
	ticket = ticket.replace('.','')
	ticket = ticket.replace('/','')
	ticket = ticket.split()
	ticket = ticket.map(lambda t: t.strip(), ticket)
	ticket = list(filter(lambda t: not t.isdigit(), ticket))
	if len(ticket) > 0:
		return ticket[0]
	else:
		return 'XXX'

ticket = pd.DataFrame(titanic.Ticket)
ticket['Ticket'] = ticket['Ticket'].map(cleanTicket)
ticket = pd.get_dummies(ticket.Ticket, prefix='Ticket')
print ticket.head()




