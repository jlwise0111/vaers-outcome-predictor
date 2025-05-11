from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd
from common import MODEL_PATH, engine


df_predict = pd.read_sql_query(f"""
select
    "SEX", "AGE_YRS", "VAX_TYPE", "VAX_DOSE_SERIES", "OUTCOME"
    from vaers_data
    where "OUTCOME" is not null
        and "SEX" is not null
        and "AGE_YRS" >= 0
        and "VAX_TYPE" is not null
        and "VAX_DOSE_SERIES" ~ '^[0-9]+$'
        and CAST("VAX_DOSE_SERIES" AS INTEGER) >= 1

""", con=engine)

df_predict['VAX_DOSE_SERIES'] = df_predict['VAX_DOSE_SERIES'].astype(int)
df_predict['AGE_YRS'] = df_predict['AGE_YRS'].astype(int)
df_predict['OUTCOME'] = df_predict['OUTCOME'].astype(str).str.strip()
df_predict.dropna(inplace=True)
df_predict = df_predict[df_predict['OUTCOME'] != '']

X = df_predict[['SEX', 'AGE_YRS', 'VAX_TYPE', 'VAX_DOSE_SERIES']]
y = df_predict['OUTCOME']

categorical = ['SEX', 'VAX_TYPE']
numeric = ['AGE_YRS', 'VAX_DOSE_SERIES']

ct = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
     ('num', 'passthrough', numeric)])

model = make_pipeline(ct, RandomForestClassifier(n_estimators=100, random_state=42))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)