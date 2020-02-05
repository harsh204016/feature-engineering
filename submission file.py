
#when you have a submission file to submit
sample = pd.read_csv("sample_submission.csv")
sample.loc[:, "target"] = y_pred
sample.to_csv("submission_using_xgboost.csv", index=False)


#when you don't have complete submission file 
from csv import DictReader
with open('test.csv') as f:
    reserve_id=[row["Accident_ID"] for row in DictReader(f)]
y_pred_list=list(y_pred)
df=pd.DataFrame(data={"Accident_ID":reserve_id,
                      "Severity":y_pred_list})
df.to_csv("second_submit.csv",sep=',',index=False)




#while working on the image dataset
test=test.values.reshape(-1,28,28,1)
pred=model.predict(test)

pred.shape

pred1=np.argmax(pred,axis=1)
pred1.shape
pred1=pred1.tolist()

l=[]
for i in range(1,28001):
    l.append(i)


df=pd.DataFrame(data={"ImageId":l,"Label":pred1})
df.to_csv("submit2.csv",sep=',',index=False)
