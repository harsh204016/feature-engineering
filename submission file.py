sample = pd.read_csv("sample_submission.csv")
sample.loc[:, "target"] = y_pred
sample.to_csv("submission_using_xgboost.csv", index=False)



from csv import DictReader
with open('test.csv') as f:
    reserve_id=[row["Accident_ID"] for row in DictReader(f)]
y_pred_list=list(y_pred)
df=pd.DataFrame(data={"Accident_ID":reserve_id,
                      "Severity":y_pred_list})
df.to_csv("second_submit.csv",sep=',',index=False)
