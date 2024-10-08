import pandas as pd
import xgboost as xgb
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('CustomerId')
    X_train = df.drop(['Exited'],axis=1)
    y_train = df[['Exited']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    xgb_mod=xgb.XGBClassifier(max_depth=3, n_estimators=50, objective='binary:logistic', seed=42, silent=True, subsample=.8)
    xgb_mod.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(xgb_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('transform_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
