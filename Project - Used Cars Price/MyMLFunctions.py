from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display
from scipy.stats import skew
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score,mean_absolute_error,mean_squared_error
from colorama import Fore, Back, Style
from random import choice


'''
Function in the module
variable_identifier
'''

def statement(statmnt):
    print(Fore.YELLOW, Back.LIGHTBLUE_EX, Style.NORMAL, statmnt, Style.RESET_ALL)

def color_chart():
    chart_color = {
        'heatmap':[
            'CMRmap','CMRmap_r','Accent','Blues','BrBG','BuGn','BuPu','Dark2','GnBu','Greens','Greys','OrRd','PRGn_r','Paired','PiYG','PuBu','PuBuGn','PuOr','PuRd',
            'Purples','RdYlGn','Set1','Set2','Set3','Spectral','YlGnBu','binary','bone_r','brg_r','coolwarm','copper_r','gist_earth_r','gist_gray_r','gist_heat_r',
            'gist_yarg','gnuplot2_r','jet','inferno_r','ocean_r','plasma_r','seismic','winter_r','twilight_shifted_r'
        ],
        'boxplot':[
            'prism','Accent','Blues','BuPu','Dark2','Greys','OrRd','PRGn','Paired','Paired_r','Pastel1','Pastel1_r','Pastel2','Pastel2_r','PiYG','PiYG_r','PuBu',
            'PuBuGn','rocket_r','PuOr','PuOr_r','PuRd','Purples','RdGy','RdYlBu','RdYlGn','Set1','Set1_r','Set2','Set2_r','Set3','Set3_r','Spectral','Wistia','YlOrBr',
            'binary','brg','bwr','cool_r','coolwarm','copper','flag_r','gist_earth','gist_ncar','gist_rainbow','gist_yarg','gnuplot','gnuplot2','gray','nipy_spectral',
            'nipy_spectral','seismic','spring','summer','tab10','tab10_r','tab20','tab20_r','tab20b','tab20b_r','tab20c','tab20c_r','twilight','twilight_shifted','winter',
        ],
        'pairplot':[
            'icefire_r','icefire','BuPu','CMRmap_r','Dark2_r','Blues','PuRd_r','Reds','Set1','Set1_r','Set2_r','autumn','binary_r','bone','brg','cool','copper','flag',
            'gist_gray','gist_heat','gist_stern_r','gnuplot2','gray_r','inferno','magma','plasma','pink','tab20c','tab20c_r','turbo','twilight_r','viridis','winter'
        ],
        'distplot':['y','r','b','grey','ocean_r','pink'],
        'countplot':[
            'Accent_r','Blues','BrBG','Dark2','Paired','RdYlGn','Set1','Set1_r','Spectral','binary','bone_r','brg_r','cividis_r','cool','copper_r','crest','cubehelix_r',
            'flag','flare','gist_earth_r','gist_gray_r','gist_heat_r','gist_ncar','gist_rainbow','gnuplot2','gnuplot','ocean_r','mako_r'
        ],
        'scatterplot':[
            'CMRmap','CMRmap_r','Dark2','Dark2_r','Paired','Paired_r','PuRd_r','RdPu','Set1','Set1_r','Set2','Set2_r','YlGnBu','YlOrBr','YlOrRd','autumn','binary','bone',
            'brg','cividis','cool','copper','crest','cubehelix','winter','viridis','twilight_shifted','twilight','tab20b_r','tab20b','tab10_r','tab10','spring','seismic',
            'rocket','rainbow','prism','plasma','ocean','mako_r','magma','inferno','icefire','hsv','hot','gnuplot','gnuplot_r','gnuplot2','gist_ncar',
            'gist_ncar','flare'
        ],
        'barplot':[
            'Accent','Blues','BrBG_r','BuPu','CMRmap_r','Dark2','Greys','PRGn_r','Paired','PiYG','PuBuGn','PuOr','PuRd','RdBu','RdGy','RdPu','RdYlBu','Set1','Set2',
            'Spectral','binary','bone_r','cool','copper_r','cubehelix_r','flag','flare','gist_heat_r','gist_ncar_r','gnuplot','gnuplot2','icefire','inferno_r','ocean_r',
            'pink_r','plasma_r','prism','prism_r','seismic_r','summer','tab10','tab20c','terrain_r','twilight','twilight_shifted','winter_r',
        ]
    }
    return chart_color

def colors():
    colors = [
        'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
        'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 
        'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
        'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
        'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
        'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone',
        'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest',
        'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
        'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
        'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 
        'inferno', 'inferno_r','jet','jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 
        'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring',
        'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 
        'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 
        'vlag_r', 'winter', 'winter_r'
            ]
    return colors

def color_terminal():
    fore = [(Fore.BLACK,'black'.upper()),(Fore.BLUE,'blue'.upper()),(Fore.CYAN,'cyan'.upper()),(Fore.GREEN,'green'.upper()),(Fore.LIGHTBLACK_EX,'lightblack'.upper()),
    (Fore.LIGHTBLUE_EX,'lightblue'.upper()),(Fore.LIGHTCYAN_EX,'lightcyan'.upper()),(Fore.LIGHTGREEN_EX,'lightgreen'.upper()),(Fore.LIGHTMAGENTA_EX,'lightmagenta'.upper()),
    (Fore.LIGHTRED_EX,'lightred'.upper()),(Fore.YELLOW,'yellow'.upper()),(Fore.WHITE,'white'.upper()),(Fore.RED,'red'.upper()),(Fore.MAGENTA,'magenta'.upper()),
    (Fore.LIGHTYELLOW_EX,'lightyellow'.upper()),(Fore.LIGHTWHITE_EX,'lightwhite'.upper())]
    back = [(Back.BLACK,'black'.upper()),(Back.BLUE,'blue'.upper()),(Back.CYAN,'cyan'.upper()),(Back.GREEN,'green'.upper()),(Back.LIGHTBLACK_EX,'lightblack'.upper()),
    (Back.LIGHTBLUE_EX,'lightblue'.upper()),(Back.LIGHTCYAN_EX,'lightcyan'.upper()),(Back.LIGHTGREEN_EX,'lightgreen'.upper()),(Back.LIGHTMAGENTA_EX,'lightmagenta'.upper()),
    (Back.LIGHTRED_EX,'lightred'.upper()),(Back.YELLOW,'yellow'.upper()),(Back.WHITE,'white'.upper()),(Back.RED,'red'.upper()),(Back.MAGENTA,'magenta'.upper()),
    (Back.LIGHTYELLOW_EX,'lightyellow'.upper()),(Back.LIGHTWHITE_EX,'lightwhite'.upper())]
    style = [
    (Style.BRIGHT,'bright'.upper()),
    (Style.DIM,'dim'.upper()),
    (Style.NORMAL,'normal'.upper()),
    ]
    for i,name1 in fore:
        for j,name2 in back:
            for k,name3 in style:
                print(i,j,k,name1,name2,name3,end='')
    Style.RESET_ALL

def variable_identifier(df,target_index=[-1],target_list=[],unique_category=20,count_plot=False):
    '''
    This is a function to classify and identity variable types and categories.
    
    This function returns all those classifications in a table.
    
    -> identification_table, column_type, column_datatype, column_category
    
    A table with all variable classification, table with all variable type(pridictor/target), table with variable data type(numeric/object),
    table with variable categories(continuous/categorical) in the order as pandas.DataFrame
    
    df : pandas DataFrame.
    
    target_index : This can be a list of prediction target column's indices.
    
    target_list : This can be a list of prediction target column's names.
                  If values are supplied this variable will supersede target_index
                  
    unique_category : This is the number of unique categorical values in given column of DataFrame to split the columns into categorical/continuous
    
    count_plot : This is a boolean fuunction that specifies if to plot a count plot of categorical columns when the function is called
                  
    Ex: x,y,z,a = variable_identifier(df)
        x
    This will return a pandas DataFrame with target being the last column in the df dataframe
    
        x,y,z,a = variable_identifier(df,target_index=[-1,3,-4])
        x
    This will return a pandas DataFrame with targets being the columns of df Dataframe
    with column index -1, 3 and -4
    
        x,y,z,a = variable_identifier(df,target_index=[10,7],target_list=['column1','column2','column3'])
        x
    This will retrun a pandas DataFrame with
    targets being the columns of df Dataframe with the column names pass in the target_list, this will ignore the indecies specified
    '''
    types_variable={}
    if len(target_list)==0:
        types_variable['Target']=list(df.columns[target_index])
    else:
        types_variable['Target']=target_list
    types_variable['Pridictor']=list(df.drop(types_variable['Target'],axis=1).columns)
    prid = pd.DataFrame(types_variable['Pridictor'],index=range(1,len(types_variable['Pridictor'])+1),columns=['PridictorVariables'])
    tar = pd.DataFrame(types_variable['Target'],index=list(range(1,len(types_variable['Target'])+1)),columns=['TargetVariables'])
    variable_type = pd.concat([prid,tar],axis=1).fillna('')

    numeric_datatypes = list(df.select_dtypes(['int','float']).columns)
    object_datatypes = list(df.select_dtypes(object).columns)
    obj = pd.DataFrame(object_datatypes,index=list(range(1,len(object_datatypes)+1)),columns=['ObjectDatatypes'])
    num = pd.DataFrame(numeric_datatypes,index=list(range(1,len(numeric_datatypes)+1)),columns=['NumericDatatypes'])
    variable_datatype = pd.concat([obj,num],axis=1).fillna('')

    cat = []
    cont = []
    title = True
    for i in df.columns:
        len_d = len(df[i].unique())
        if len_d<=unique_category:
            cat.append(f'{i}=>{len_d}')
            if count_plot==True:
                if title==True:
                    title=False
                    print(Fore.MAGENTA,Back.YELLOW,Style.BRIGHT,f'{" "*29}categories{" "*27}'.upper(),Style.RESET_ALL)
                if len(df[i].unique())<=25:
                    plt.figure(figsize=(10,10))
                    sns.countplot(df[i], palette=choice(color_chart()['countplot'])).set_title(i,fontsize=20)
                    plt.show()
        else:
            cont.append(i)
    cat = pd.DataFrame(cat,index=range(1,len(cat)+1),columns=['CategoricalVariables'])
    cont = pd.DataFrame(cont,index=range(1,len(cont)+1),columns=['ContinuousVariables'])
    variable_category = pd.concat([cat,cont],axis=1).fillna('')
    
    variable_identification = pd.concat([variable_type,variable_datatype,variable_category],axis=1).fillna('')
    return variable_identification, variable_type, variable_datatype, variable_category

def missingdata(df,drop_na=3.0,fill_na=40.0,plotChart=True,getTable=False):
    '''
    This Function return a DataFrame and plots the DataFrame. This DataFrame consists of all the columns which have missing values and their percent
    
    df : pandas DataFrame
    '''
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    name = dict(zip(range(1,len(total)+1),total.keys()))
    name = pd.DataFrame(name.values(),index=name.keys(),columns=['ColumnName'])
    total = dict(zip(range(1,len(total)+1),total.values))
    total = pd.DataFrame(total.values(),index=total.keys(),columns=['TotalMissingValues'])
    percent = dict(zip(range(1,len(percent)+1),percent.values))
    percent = pd.DataFrame(percent.values(),index=percent.keys(),columns=['PercentageMissing'])
    md=pd.concat([name,total, percent], axis=1)
    md= md[md.PercentageMissing > 0]
    try:
        if plotChart==True:
            plt.figure(figsize=(13,6))
            plt.xticks(rotation='75')
            sns.barplot(md.ColumnName, md.PercentageMissing,color="blue",alpha=0.8)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Percent of missing values', fontsize=15)
            plt.title('Missing data% by feature', fontsize=25)
            plt.show()
        else:
            pass
        na = {}
        clean = {}
        drop = {}
        drop_columns=100.0-fill_na
        for i in md.index:
            per = md.loc[i][2]
            if per<=drop_na:
                na[md.ColumnName[i]]=per
            elif per<=fill_na:
                clean[md.ColumnName[i]]=per
            else:
                drop[md.ColumnName[i]]=per
        d = pd.concat(
            [pd.DataFrame(na.keys(),index=range(1,len(na)+1),columns=['DropNa']),
             pd.DataFrame(clean.keys(),index=range(1,len(clean)+1),columns=['FillNa']),
             pd.DataFrame(drop.keys(),index=range(1,len(drop)+1),columns=['DropColumn']),],axis=1)
        criteria = {
            1:f'''Drop Rows When missing data is Less Than {drop_na}%'''.title(),
            2:f'Fill na When missing data is Less Than {fill_na}%'.title(),
            3:f'Drop Columns When missing data is More Than {drop_columns}%'.title(),
        }
        criteria = pd.DataFrame(criteria.values(),index=criteria.keys(),columns=['Criteria'])
        md = pd.concat([md,d,criteria],axis=1).fillna('')
        if getTable==True:
            display(md)
            return md
    except ValueError:
        return print(Fore.YELLOW,Back.LIGHTBLUE_EX,'No Missing Values')

    
def skewness(columns,df):
    sns.set_color_codes()
    for i in columns:
        print(Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f"{i} = {skew(df[i])}",Style.RESET_ALL)
        sns.distplot(df[i],color='r')
        plt.axvline(df[i].mean(), color="green")
        plt.axvline(df[i].median(), color="red")
        plt.show()
        print("---------------------------------------------------------------------------------")
        

def skewness_cbrt_method(columns,df):
    '''
    Returns a distribution plot of the columns specified in the columns while applying cube root to try to fix skewness if any
    
    columns : List of columns to be plotted
    
    df : Dataframe to be processed
    '''
    df1=df
    df2=df
    sns.set_color_codes()
    for i in columns:
        df1[i]=np.sqrt(df[i])
        df2[i]=np.cbrt(df[i])
        print(f'{i} => {skew((df[i]))} => {skew(np.sqrt(df[i]))} => {skew(np.cbrt(df[i]))}')
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        sns.distplot(df[i],color='r')
        plt.axvline(df[i].mean(), color="green")
        plt.axvline(df[i].median(), color="red")
        plt.title(i,fontsize=20)
        plt.subplot(1,3,2)
        sns.distplot(df1[i],color='r')
        plt.axvline(df1[i].mean(), color="green")
        plt.axvline(df1[i].median(), color="red")
        plt.title(f'{i} Sqrt',fontsize=20)
        plt.subplot(1,3,3)
        sns.distplot(df2[i],color='r')
        plt.axvline(df2[i].mean(), color="green")
        plt.axvline(df2[i].median(), color="red")
        plt.title(f'{i} Cbrt',fontsize=20)
        plt.show()
        print(f'{"-"*100}')
        
def count(df,columns):
    '''
    
    '''
    for cols in columns:
        plt.figure(figsize=(6,4))
        plt.xticks(rotation='75')
        sns.countplot(df[cols],color="blue",alpha=0.8)
        plt.xlabel('Categories', fontsize=15)
        plt.ylabel('Count of values', fontsize=15)
        plt.title(f'Categorical Distribution of {cols}', fontsize=15)
        plt.show()
        
def model_accuracy(model, data, model_report=True, models_obj=False):
    '''
    Retuns a accuracy, mean absolute error, mean squared error and root mean squared error of the model object passed in 'model'
    
    model : It is the object of the model to be tested
    
    data : It should be a list of xtrain, ytrain, xtest, ytest
    
    models_obj : Boolean value determines if to return the model and predicted value
    '''
    xtrain = data[0]
    ytrain = data[2]
    xtest = data[1]
    ytest = data[3]
    model_return = model
    start = time()
    print('fitting')
    model.fit(xtrain, ytrain)
    end = time()
    print(end-start)
    start = time()
    print('predicting xtest')
    ypred = model.predict(xtest)
    end = time()
    print(end-start)
    start = time()
    print('predicting xtrain')
    ytrain1 = model.predict(xtrain)
    end = time()
    print(end-start)
    start = time()
    print('accuracy')
    mae = mean_absolute_error(ytest,ypred)
    mse = mean_squared_error(ytest,ypred)
    rmse = np.sqrt(mean_squared_error(ytest,ypred))
    test_ac = r2_score(ytest,ypred)*100
    train_ac = r2_score(ytrain,ytrain1)*100
    end = time()
    print(end-start)
    if model_report==True:
        print(" Training Accuracy      :",Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f'{train_ac}%',Style.RESET_ALL)
        print(" Test Accuracy          :",Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f'{test_ac}%',Style.RESET_ALL)
        print("Mean Absolute Error     :",Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f'{mae}',Style.RESET_ALL)
        print("Mean Squared Error      :",Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f'{mse}',Style.RESET_ALL)
        print("Root Mean Squared Error :",Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f'{rmse}',Style.RESET_ALL)
    if models_obj == True:
        return model_return, ypred, ytrain1, test_ac, train_ac
    else:
        return None


def model_classification_report(model, data, print_report=True, models_obj=False):
    '''
    Retuns a accuracy, confusion matrix and classification report of the model object passed in 'model'
    return model_return, ypred, ypred1, ac, ac1
    
    model : It is the object of the model to be tested
    
    data : It should be a list of [xtrain, ytrain, xtest, ytest]
    
    models_obj : Boolean value determines if to return the model and predicted value
    '''
    xtrain = data[0]
    ytrain = data[2]
    xtest = data[1]
    ytest = data[3]
    model_return = model
    start = time()
    print('fitting')
    model.fit(xtrain, ytrain)
    end = time()
    print(end-start)
    start = time()
    print('predicting xtest')
    ypred = model.predict(xtest)
    end = time()
    print(end-start)
    start = time()
    print('predicting xtrain')
    ypred1 = model.predict(xtrain)
    end = time()
    print(end-start)
    start = time()
    cr = classification_report(ytest, ypred)
    cm = confusion_matrix(ytest,ypred)
    test_ac = accuracy_score(ytest,ypred)*100
    train_ac = accuracy_score(ytrain,ypred1)*100
    if print_report==True:
        print(f"Training Accuracy : ",Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f"{train_ac}%",Style.RESET_ALL,f"\nTest Accuracy :     ",Fore.RED,Back.LIGHTYELLOW_EX,Style.BRIGHT,f"{test_ac}%",Style.RESET_ALL,f"\n\nConfusion Matrix : \n{cm}\n\nClassification Report : \n{cr}")
        end = time()
        print(end-start)
    if models_obj == True:
        return model_return, ypred, ytrain1, test_ac, train_ac
    else:
        return None