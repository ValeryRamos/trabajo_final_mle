import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Definición de los pasos para la transformación de datos

# Transformadores numéricos
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputar valores faltantes con la media
    ('scaler', StandardScaler())  # Estandarizar características numéricas
])

# Transformadores categóricos
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputar valores faltantes
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Codificación One-Hot
])

# Preprocesador combinado
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)).set_index('ID')
    print(filename, 'cargado correctamente')
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    # Aplicamos las transformaciones específicas
    df = preprocessor.fit_transform(df)
    return pd.DataFrame(df)

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename), index=False)
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('defaultcc.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, tdf1.columns.tolist() + ['Exited'], 'transform_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('defaultcc_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, tdf2.columns.tolist() + ['Existed'], 'transform_val.csv')
    
    # Matriz de Scoring
    df3 = read_file_csv('defaultcc_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, tdf3.columns.tolist(), 'transform_score.csv')


if __name__ == "__main__":
    main()

