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
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # Codificación One-Hot
])

# Preprocesador combinado
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)).set_index('CustomerId')
    print(filename, 'cargado correctamente')
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    # Verificar si la columna 'Exited' está presente
    target_present = 'Exited' in df.columns
    if target_present:
        y = df['Exited']  # Mantener la columna 'Exited' como target
        X = df.drop(columns=['Exited'])  # Eliminar la columna 'Exited' del DataFrame de características
    else:
        X = df
    
    # Aplicar las transformaciones al DataFrame de características
    X_transformed = preprocessor.fit_transform(X)
    
    # Obtener nombres de columnas resultantes
    numeric_cols = numeric_features
    categorical_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_cols = numeric_cols + list(categorical_cols)
    
    # Crear el DataFrame con los nombres de columnas y el índice original
    df_transformed = pd.DataFrame(X_transformed, columns=all_cols, index=X.index)
    
    # Volver a añadir la columna del target si está presente
    if target_present:
        df_transformed['Exited'] = y
        df_transformed = df_transformed[['Exited'] + [col for col in df_transformed.columns if col != 'Exited']]  # Asegura que 'Exited' esté al inicio
    
    return df_transformed

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, filename):
    df.to_csv(os.path.join('../data/processed/', filename), index=True)
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('train.csv')
    tdf1 = data_preparation(df1)
    # Incluye 'Exited' al exportar
    data_exporting(tdf1, 'transform_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('test.csv')
    tdf2 = data_preparation(df2)
    # No incluye 'Exited' al exportar
    data_exporting(tdf2, 'transform_test.csv')
    
    # Matriz de Scoring
    df3 = read_file_csv('scorear.csv')
    tdf3 = data_preparation(df3)
    # No incluye 'Exited' al exportar
    data_exporting(tdf3, 'transform_score.csv')

if __name__ == "__main__":
    main()

