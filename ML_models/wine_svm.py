# 🍷 Wine Quality - Support Vector Machine
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")
df=pd.read_csv("wine.csv")

df=df.dropna(subset=["quality"])
df.fillna(df.median(numeric_only=True), inplace=True)
if "wine_type" in df.columns:
    df["wine_type"]=LabelEncoder().fit_transform(df["wine_type"].astype(str))

df["total_acidity"]=df["fixed acidity"]+df["volatile acidity"]+df["citric acid"]
df["sulfur_ratio"]=df["free sulfur dioxide"]/df["total sulfur dioxide"].replace(0,np.nan)
df["sugar_per_acid"]=df["residual sugar"]/(df["total_acidity"]+1e-5)
df["acid_sugar_ratio"]=df["total_acidity"]/(df["residual sugar"]+1e-5)
df["density_alcohol_ratio"]=df["density"]/(df["alcohol"]+1e-5)
df["is_high_alcohol"]=(df["alcohol"]>df["alcohol"].median()).astype(int)
df["is_high_sugar"]=(df["residual sugar"]>df["residual sugar"].median()).astype(int)
df.replace([np.inf,-np.inf],np.nan,inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

def label(q): return 0 if q<=5 else (1 if q==6 else 2)
df["quality_class"]=df["quality"].apply(label)
X=df.drop(columns=["quality","quality_class"]); y=df["quality_class"]

X=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.25,random_state=42)

model=SVC(kernel="rbf",C=2,gamma="scale",random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("=== Support Vector Machine ===")
print("Accuracy:", round(accuracy_score(y_test,y_pred),4))
print(classification_report(y_test,y_pred,target_names=["Low","Medium","High"]))
