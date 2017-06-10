import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd 
from matplotlib import style
import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import Imputer , Normalizer , scale
# from sklearn.cross_validation import train_test_split , StratifiedKFold
# from sklearn.feature_selection import RFECV
# from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

style.use('ggplot')
# sns.set(style = 'white')
# pylab.rcParams['figure.figsize'] = 8, 6



# ######## Helper Functions for Visualization
# def plot_correlation_map(df):
# 	corr = df.corr()
# 	f, ax = plt.subplots(figsize = (12,10))
# 	cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
# 	_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = { 'fontsize' : 12 })
# 	plt.show()

# def plot_distribution( df , var , target , **kwargs ):
#     row = kwargs.get( 'row' , None )
#     col = kwargs.get( 'col' , None )
#     facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
#     facet.map( sns.kdeplot , var , shade= True )
#     facet.set( xlim=( 0 , df[ var ].max() ) )
#     facet.add_legend()
#     plt.show()

# def plot_categories( df , cat , target , **kwargs ):
#     row = kwargs.get( 'row' , None )
#     col = kwargs.get( 'col' , None )
#     facet = sns.FacetGrid( df , row = row , col = col )
#     facet.map( sns.barplot , cat , target )
#     facet.add_legend()
#     plt.show()

# def plot_variable_importance( X , y ):
#     tree = DecisionTreeClassifier( random_state = 99 )
#     tree.fit( X , y )
#     plot_model_var_imp( tree , X , y )


# def plot_model_var_imp( model , X , y ):
#     imp = pd.DataFrame( 
#         model.feature_importances_  , 
#         columns = [ 'Importance' ] , 
#         index = X.columns 
#     )
#     imp = imp.sort_values( [ 'Importance' ] , ascending = True )
#     imp[ : 10 ].plot( kind = 'barh' )
#     print (model.score( X , y ))
#     plt.show()
   

def status(feature):
    print 'Processing ' + feature + '.. OK!'


def combined_data():

    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")

    PassengerId = test['PassengerId']
    survived = train['Survived']

    train.drop('Survived', 1, inplace = True)

    combined = train.append(test, ignore_index = True)  # Make full data
    combined.drop('Ticket', 1, inplace = True)

    # Get the data to work on.. containing 891 elements
    # titanic = combined[:891]
    return combined


combined = combined_data()


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


########## Creating labels '1's and '0's
def get_titles():

    global combined

    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    
    Title_Dictionary = {
                        "Capt"			:	"Officer",
                        "Col"			:	"Officer",
                        "Major"			:	"Officer",
                        "Jonkheer"		:   "Royalty",
                        "Don"			:	"Royalty",
                        "Sir" 			:	"Royalty",
                        "Dr"			:	"Officer",
                        "Rev"			:	"Officer",
                        "the Countess"	:	"Royalty",
                        "Dona"			:	"Royalty",
                        "Mme"			:	"Mrs",
                        "Mlle"			:	"Miss",
                        "Ms"			:	"Mrs",
                        "Mr" 			:	"Mr",
                        "Mrs" 			:	"Mrs",
                        "Miss" 			:	"Miss",
                        "Master" 		:	"Master",
                        "Lady" 			:	"Royalty"
                    }

    combined['Title'] = combined.Title.map(Title_Dictionary)
    # title = pd.get_dummies(title.Title)
    # print title.head()


get_titles()

def process_name():

    global combined

    combined.drop('Name', 1, inplace = True)

    title_dummies = pd.get_dummies(combined['Title'], prefix = 'Title')
    combined = pd.concat([combined, title_dummies], axis = 1)

    combined.drop('Title', 1, inplace = True)

    status('Names')


def process_sex():

    global combined
    combined['Sex'] = combined['Sex'].map({'female':0, 'male':1})

    status('Sex')


process_sex()


# ########## Processing Age is not working at 'apply' method..

# grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])
# grouped_median_train = grouped_train.median()

# grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])
# grouped_median_test = grouped_test.median()

# print grouped_median_train
# print grouped_median_test
# print grouped_median_train.loc[0,1,'Miss']['Age']

# def process_age():

#     global combined
#     def fillAges(row, median):
#         if row['Sex'] == 0 and row['Pclass'] == 1:
#             if row['Title'] == 'Miss':
#                 return median.loc[0,1,'Miss']['Age']
#             elif row['Title'] == 'Mrs':
#                 return median.loc[0,1,'Mrs']['Age']
#             elif row['Title'] == 'Officer':
#                 return median.loc[0,1,'Officer']['Age']
#             elif row['Title'] == 'Royalty':
#                 return median.loc[0,1,'Royalty']['Age']
#         elif row['Sex'] == 0 and row['Pclass'] == 2:
#             if row['Title'] == 'Miss':
#                 return median.loc[0,2,'Miss']['Age']
#             elif row['Title'] == 'Mrs':
#                 return median.loc[0,2,'Mrs']['Age']
#             elif row['Title'] == 'Officer':
#                 return median.loc[0,2,'Officer']['Age']
#             elif row['Title'] == 'Royalty':
#                 return median.loc[0,2,'Royalty']['Age']
#         elif row['Sex'] == 0 and row['Pclass'] == 3:
#             if row['Title'] == 'Miss':
#                 return median.loc[0,3,'Miss']['Age']
#             elif row['Title'] == 'Mrs':
#                 return median.loc[0,3,'Mrs']['Age']
#             elif row['Title'] == 'Officer':
#                 return median.loc[0,3,'Officer']['Age']
#             elif row['Title'] == 'Royalty':
#                 return median.loc[0,3,'Royalty']['Age']
#         elif row['Sex'] == 1 and row['Pclass'] == 1:
#             if row['Title'] == 'Miss':
#                 return median.loc[1,1,'Miss']['Age']
#             elif row['Title'] == 'Mrs':
#                 return median.loc[1,1,'Mrs']['Age']
#             elif row['Title'] == 'Officer':
#                 return median.loc[1,1,'Officer']['Age']
#             elif row['Title'] == 'Royalty':
#                 return median.loc[1,1,'Royalty']['Age']
#         elif row['Sex'] == 1 and row['Pclass'] == 3:
#             if row['Title'] == 'Miss':
#                 return median.loc[1,2,'Miss']['Age']
#             elif row['Title'] == 'Mrs':
#                 return median.loc[1,2,'Mrs']['Age']
#             elif row['Title'] == 'Officer':
#                 return median.loc[1,2,'Officer']['Age']
#             elif row['Title'] == 'Royalty':
#                 return median.loc[1,2,'Royalty']['Age']
#         elif row['Sex'] == 1 and row['Pclass'] == 3:
#             if row['Title'] == 'Miss':
#                 return median.loc[1,3,'Miss']['Age']
#             elif row['Title'] == 'Mrs':
#                 return median.loc[1,3,'Mrs']['Age']
#             elif row['Title'] == 'Officer':
#                 return median.loc[1,3,'Officer']['Age']
#             elif row['Title'] == 'Royalty':
#                 return median.loc[1,3,'Royalty']['Age']

    
#     # k = 0
#     # print combined.Age.iloc[5]
#     # while k < 891:
#     #     if np.isnan(combined.Age.iloc[k]).all():
#     #         combined.Age.iloc[k] = fillAges(combined.iloc[[k]] , grouped_median_train)
#     #     k += 1
#     # print combined.Age.iloc[5]
#     # while k >= 891 and k < 1309:
#     #     if np.isnan(combined.Age.iloc[k]).all():
#     #         combined.Age.iloc[k] = fillAges(combined.iloc[[k]] , grouped_median_test)
#     #     k += 1


#     print combined.Age.iloc[5]
#     combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age'])
#                                                   else r['Age'], axis=1)
#     print combined.Age.iloc[5]
#     combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age'])
#                                                   else r['Age'], axis=1)


#     status('Age')




# # combined = combined_data()
# # get_titles()
# # process_sex()

# process_age()

# #############

def process_age():

    global combined
    combined.head(891).Age.fillna(combined.head(891).Age.mean(), inplace = True)
    combined.iloc[891:].Age.fillna(combined.iloc[891:].Age.mean(), inplace = True)

    status('Age')


process_age()




process_name()


def process_fare():

    global combined
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace = True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace = True)

    status('Fares')


process_fare()



def process_embarked():

    global combined
    # doesnt show much correlation.. so replace with the most frequent entry 'S'
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix = 'Embarked')
    combined = pd.concat([combined, embarked_dummies], axis = 1)

    combined.drop('Embarked', axis = 1, inplace = True)

    status('Embarked')


process_embarked()




def process_cabin():

    global combined

    combined.Cabin.fillna('U', inplace=True)
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix = 'Cabin')

    combined = pd.concat([combined, cabin_dummies], axis = 1)
    combined.drop('Cabin', axis = 1, inplace = True)

    status('Cabin')


process_cabin()



def process_pclass():
    
    global combined
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    combined = pd.concat([combined, pclass_dummies], axis = 1)    
    combined.drop('Pclass', axis = 1, inplace = True)
    
    status('Pclass')


process_pclass()



def process_family():
    
    global combined
    # New feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)
    
    status('Family')


process_family()



passenger_id = combined.iloc[891:].PassengerId
combined.drop('PassengerId', inplace = True, axis = 1)

print combined.Age

###############    #######
def compute_cross_val_score(clf, X, y, scoring = 'accuracy'):

    score = cross_val_score(clf, X, y, cv = 5, scoring=scoring) 
    return np.mean(score)


def recover_test_train_from_combined():

    global combined

    train_ = pd.read_csv('input/train.csv')
    train_.drop('Ticket', 1, inplace = True)
    
    survived = train_.Survived
    train = combined.head(891)
    test = combined.iloc[891:]

    return train, test, survived



train, test, survived = recover_test_train_from_combined()


clf = RandomForestClassifier(n_estimators = 50, max_features = "sqrt")
clf.fit(train, survived)



features = pd.DataFrame()

features['Features'] = train.columns
features['Importance'] = clf.feature_importances_

features.sort_values(by = ['Importance'], ascending = True, inplace = True)
features.set_index('Features', inplace = True)
features.plot(kind='barh', figsize=(20, 20))
# plt.show()



model = SelectFromModel(clf, prefit = True)
train_reduced = model.transform(train)
print train_reduced.shape
test_reduced = model.transform(test)
print test_reduced.shape


# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [1, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, survived)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, survived)


output = model.predict(test).astype(int)
df_output = pd.DataFrame()
test_ = pd.read_csv('input/test.csv')
df_output['PassengerId'] = test_['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output/output.csv',index=False)



##################      ######



# sex_series = pd.Series(np.where(combined.Sex=='male', 1, 0), name='Sex')
# sex = pd.Series.to_frame(sex_series)
# # print sex.head()

# embarked = pd.get_dummies(combined.Embarked, prefix='Embarked')
# # print embarked.head()

# Pclass = pd.get_dummies(combined.Pclass, prefix='Pclass')
# # print Pclass.head()
# # embarked = pd.get_dummies(titanic.embarked, prefix='Embarked')
# # embarked = pd.get_dummies(titanic.embarked, prefix='Embarked')
# # embarked = pd.get_dummies(titanic.embarked, prefix='Embarked')
# ##########

# combined.Age.fillna(combined.Age.mean(), inplace = True)
# combined.Fare.fillna(combined.Fare.mean(), inplace = True)
# # print titanic.head()

# combined.Cabin.fillna('U',inplace=True)
# cabin = pd.DataFrame(combined.Cabin)
# cabin['Cabin'] = cabin['Cabin'].map(lambda c : c[0])
# cabin = pd.get_dummies(cabin.Cabin, prefix='Cabin')
# # print cabin.head()

# ## new feature introduced -- family
# family = pd.DataFrame()
# family['Family_Size'] = combined['Parch'] + combined['SibSp'] + 1
# family['Family_Single'] = family['Family_Size'].map(lambda s : 1 if s == 1 else 0)
# family['Family_Small']  = family['Family_Size'].map(lambda s : 1 if 2 <= s <= 4 else 0)
# family['Family_Large']  = family['Family_Size'].map(lambda s : 1 if 5 <= s else 0)
# # print family.head()

# titanic_X = pd.concat([combined.Age, combined.Fare, cabin, embarked, family, Pclass, sex], axis=1)
# # print titanic.describe()
# # titanic.dropna(inplace = True)
# train_valid_X = titanic_X[:891]
# train_valid_Y = titanic.Survived
# print train_valid_Y.head()
# # titanic_raw.dropna(inplace = True)
# test_X = titanic_X[891:]
# print test_X.head()
# print (titanic_X.shape , test_X.shape, train_valid_X.shape, train_valid_Y.shape)

# train_X, valid_X, train_Y, valid_Y = train_test_split(train_valid_X, train_valid_Y, train_size = 0.7)
# print (titanic_X.shape , train_X.shape , valid_X.shape , train_Y.shape , valid_Y.shape , test_X.shape)

# plot_variable_importance(train_X, train_Y)

# # tree = DecisionTreeClassifier( random_state = 99 )
# # tree.fit( train_X , train_Y )
# # rfecv = RFECV( estimator = tree , step = 1 , cv = StratifiedKFold( train_Y , 2 ) , scoring = 'accuracy' )
# # rfecv.fit( train_X , train_Y )
# # print (rfecv.score( train_X , train_Y ) , rfecv.score( valid_X , valid_Y ))
# # print( "Optimal number of features : %d" % rfecv.n_features_ )
# # ## Plot number of features VS. cross-validation scores
# # plt.figure()
# # plt.xlabel( "Number of features selected" )
# # plt.ylabel( "Cross validation score (nb of correct classifications)" )
# # plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
# # plt.show()

# model = DecisionTreeClassifier( random_state = 99 )
# model.fit( train_X , train_Y )
# test_Y = model.predict( test_X )
# passenger_id = combined[891:].PassengerId
# test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
# test.shape
# test.head()
# test.to_csv( 'output/titanic_pred.csv' , index = False )

