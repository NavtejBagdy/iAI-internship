from transformers import pipeline
import torch

import pandas as pd

def create_dataframe(data):
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df
tqa = pipeline(task="table-question-answering",
               model="google/tapas-base-finetuned-wtq")

data =[
    ["sno", "date", "name", "job"],
    [1, "05/12/95", "John", "engineer"],
    [2, "08/23/90", "Emily", "manager"],
    [3, "02/17/88", "Michael", "analyst"],
    [4, "09/30/91", "Sophia", "programmer"],
    [5, "06/08/87", "Olivia", "consultant"],
    [6, "11/15/92", "Ethan", "technician"],
    [7, "03/29/89", "Ava", "sales"],
    [8, "10/02/94", "Jacob", "finance"],
    [9, "04/07/82", "Mia", "teacher"],
    [10, "01/14/79", "Noah", "coordinator"],
    [11, "07/21/84", "Isabella", "assistant"],
    [12, "12/04/91", "William", "administrator"],
    [13, "03/17/85", "James", "programmer"],
    [14, "09/09/93", "Sophia", "specialist"],
    [15, "05/23/88", "Harper", "supervisor"],
    [16, "01/26/94", "Benjamin", "programmer"],
    [17, "08/10/80", "Grace", "assistant"],
    [18, "11/02/89", "Liam", "technician"],
    [19, "06/25/83", "Emma", "manager"],
    [20, "02/08/92", "Charlotte", "programmer"],
    [21, "10/10/77", "Noah", "engineer"],
    [22, "09/01/95", "Ava", "developer"],
    [23, "07/12/81", "Daniel", "analyst"],
    [24, "03/04/96", "Emily", "specialist"],
    [25, "06/27/87", "Michael", "supervisor"],
    [26, "11/08/93", "Lily", "programmer"],
    [27, "04/19/90", "Ethan", "technician"],
    [28, "09/02/95", "Amelia", "sales"],
    [29, "12/13/83", "William", "finance"],
    [30, "01/16/89", "Harper", "teacher"],
    [31, "07/31/75", "Mia", "coordinator"],
    [32, "05/06/94", "Oliver", "assistant"],
    [33, "08/21/92", "Emily", "administrator"],
    [34, "03/10/86", "Jacob", "programmer"],
    [35, "10/29/93", "Ava", "specialist"],
    [36, "06/14/88", "Sophia", "supervisor"],
    [37, "01/18/95", "Benjamin", "programmer"],
    [38, "09/14/80", "Grace", "assistant"],
    [39, "11/27/89", "Liam", "technician"],
    [40, "06/01/82", "Emma", "manager"],
    [41, "02/12/91", "Charlotte", "programmer"],
    [42, "10/16/77", "Noah", "engineer"],
    [43, "09/03/95", "Ava", "developer"],
    [44, "07/06/81", "Daniel", "analyst"],
    [45, "04/20/96", "Emily", "specialist"],
    [46, "05/02/87", "Michael", "supervisor"],
    [47, "11/11/93", "Lily", "programmer"],
    [48, "04/22/90", "Ethan", "technician"],
    [49, "08/28/95", "Amelia", "sales"],
    [50, "12/07/83", "William", "finance"]
]
df=create_dataframe(data)
df = df.astype(str)

query = "Job name of William?"
print(tqa(table=df, query=query)["answer"])

query = "All names belong to programmer?"
print(tqa(table=df, query=query)["answer"])

query = "all name with similar job?"
print(tqa(table=df, query=query)["answer"])

query = "serial numbers of emily?"
print(tqa(table=df, query=query)["answer"])

query = "Joining Date of emily?"
print(tqa(table=df, query=query)["answer"])

df1 = [
    ["emp_id", "emp_name", "joining_date", "ending_date", "date_of_birth", "insurance_renewal_date"],
    ["emp_001", "Jack", "12/3/21", "12/3/23", "12/4/1996", "12/4/23"],
    ["emp_002", "Ria", "12/3/21", "_ (nil)", "12/4/1996", "12/4/23"],
    ["emp_003", "John", "12/4/21", "12/4/23", "11/6/1990", "11/6/23"],
    ["emp_004", "Emma", "12/5/21", "12/5/23", "9/10/1988", "9/10/23"],
    ["emp_005", "Alex", "12/6/21", "_ (nil)", "5/17/1992", "5/17/23"],
    ["emp_006", "Lily", "12/7/21", "12/7/23", "7/22/1995", "7/22/23"],
    ["emp_007", "Max", "12/8/21", "12/8/23", "3/30/1991", "3/30/23"],
    ["emp_008", "Sophia", "12/9/21", "_ (nil)", "8/12/1993", "8/12/23"],
    ["emp_009", "Ethan", "12/10/21", "12/10/23", "1/8/1997", "1/8/23"],
    ["emp_010", "Olivia", "12/11/21", "_ (nil)", "4/21/1989", "4/21/23"]
]

df1=create_dataframe(df1)
df1=df1.astype(str)



query = "emp_id of Ria?"
print(tqa(table=df1, query=query)["answer"])

query = "ending_date of Ria?"
print(tqa(table=df1, query=query)["answer"])

query = "date of birth of Ria?"
print(tqa(table=df1, query=query)["answer"])

query = "all employees with ending_date as nil except ria and alex?"
print(tqa(table=df1, query=query)["answer"])

