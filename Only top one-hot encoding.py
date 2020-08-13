#get the top 10 value from X2 column
top_10 = [ x for x in data.X2.value_counts().sort_values(ascending=False).head(10).index]

#and now make the 10 binary variables
for label in top_10:
  data[label] = np.where(data['X2']==label , 1, 0)
  
#view the results 
data[['X2]+top_10].head(20)

#function 
def one_hot_top(df,variable,top_x_labels):
  
  for label in top_x_labels:
    df[variable+'_'+label] = np.where(data[variable]==label,1,0)
    
  return df
    
data = pd.read_csv('data.csv',usecols=['X1',X2'])

one_hot_top_x(data, 'X2' , top_10)
data.head()
