a = "age,gender,maritalstatus,education,occupation,annualincome,homeowner,state,monthlybilledamount,totalminsusedinlastmonth,unpaidbalance,numberofmonthunpaid,numberofcomplaints,numdayscontractequipmentplanexpiring,penaltytoswitch,calldroprate,callfailurerate,percentagecalloutsidenetwork,totalcallduration_mean,totalcallduration_std,totalcallduration_min,totalcallduration_max,avgcallduration_mean,avgcallduration_std,avgcallduration_min,avgcallduration_max,true_churn,pred_churn,probability_churn"
b="66,Male,Married,High School or below,Non-technology Related Job,76475,Yes,CO,94,227,186,1,2,1,405,0.06,0.03,0.33,607.5,485.78235867515815,264,951,290.5,37.476659402887016,264,317,0,0,0.004831951572898782"

aa = a.split(",")
bb = b.split(",")

for i in range(len(aa)):
    print(f'"{aa[i]}": {bb[i]},')