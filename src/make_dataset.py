mport pandas as pd
import numpy as np

def read_file_csv(filename):
    """Función para leer el archivo CSV y cargarlo en un DataFrame"""
    df = pd.read_csv(filename)
    print(f'{filename} cargado correctamente')
    return df


def transform_categorical(df):
    """
    Transforma variables categóricas según el diccionario de datos.
    Convierte categorías en valores numéricos o aplica codificación.
    """
    # MSSubClass: mapeo a las descripciones de tipo de vivienda
    MSSubClass_map = {
        20: '1-STORY 1946 & NEWER ALL STYLES', 30: '1-STORY 1945 & OLDER',
        40: '1-STORY W/FINISHED ATTIC ALL AGES', 45: '1-1/2 STORY - UNFINISHED ALL AGES',
        50: '1-1/2 STORY FINISHED ALL AGES', 60: '2-STORY 1946 & NEWER',
        70: '2-STORY 1945 & OLDER', 75: '2-1/2 STORY ALL AGES',
        80: 'SPLIT OR MULTI-LEVEL', 85: 'SPLIT FOYER', 90: 'DUPLEX - ALL STYLES AND AGES',
        120: '1-STORY PUD 1946 & NEWER', 150: '1-1/2 STORY PUD ALL AGES',
        160: '2-STORY PUD 1946 & NEWER', 180: 'PUD MULTILEVEL',
        190: '2 FAMILY CONVERSION ALL STYLES AND AGES'
    }
    df['MSSubClass'] = df['MSSubClass'].map(MSSubClass_map)

    # MSZoning: mapeo a sus valores categóricos
    MSZoning_map = {
        'A': 'Agriculture', 'C': 'Commercial', 'FV': 'Floating Village Residential',
        'I': 'Industrial', 'RH': 'Residential High Density', 'RL': 'Residential Low Density',
        'RP': 'Residential Low Density Park', 'RM': 'Residential Medium Density'
    }
    df['MSZoning'] = df['MSZoning'].map(MSZoning_map)

    # Street: Pave or Grvl
    Street_map = {'Grvl': 'Gravel', 'Pave': 'Paved'}
    df['Street'] = df['Street'].map(Street_map)

    # Alley: Convertimos 'NA' en una categoría 'No Alley Access'
    Alley_map = {'Grvl': 'Gravel', 'Pave': 'Paved', 'NA': 'No alley access'}
    df['Alley'] = df['Alley'].fillna('NA').map(Alley_map)

    # LotShape
    LotShape_map = {'Reg': 'Regular', 'IR1': 'Slightly irregular', 'IR2': 'Moderately Irregular', 'IR3': 'Irregular'}
    df['LotShape'] = df['LotShape'].map(LotShape_map)

    # LotConfig
    LotConfig_map = {'Inside': 'Inside', 'Corner': 'Corner', 'CulDSac': 'Cul-de-sac', 'FR2': 'Frontage 2 sides', 'FR3': 'Frontage 3 sides'}
    df['LotConfig'] = df['LotConfig'].map(LotConfig_map)

    # LandContour
    LandContour_map = {'Lvl': 'Level', 'Bnk': 'Banked', 'HLS': 'Hillside', 'Low': 'Depression'}
    df['LandContour'] = df['LandContour'].map(LandContour_map)

    # Utilities
    Utilities_map = {'AllPub': 'All public utilities', 'NoSewr': 'No Sewer', 'NoSeWa': 'No Sewer or Water', 'ELO': 'Electricity Only'}
    df['Utilities'] = df['Utilities'].map(Utilities_map)

    return df


def handle_missing_values(df):
    """Maneja los valores faltantes o 'NA' en el dataset"""
    # Transformamos 'NA' en valores categóricos donde tiene sentido
    df['Alley'] = df['Alley'].fillna('NA')
    df['BsmtQual'] = df['BsmtQual'].fillna('NA')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
    df['GarageFinish'] = df['GarageFinish'].fillna('NA')
    
    # Imputamos valores faltantes en columnas numéricas como LotFrontage
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

    return df


def data_preparation(filename):
    """Función principal para la preparación de los datos"""
    # Cargamos el dataset
    df = read_file_csv(filename)

    # Mapeamos las columnas categóricas a valores legibles
    df = transform_categorical(df)

    # Manejo de valores faltantes
    df = handle_missing_values(df)

    # Devolvemos el dataframe limpio
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('train.csv')
    tdf1 = data_preparation(df1)
    featuresdf1 = tdf1.columns
    data_exporting(tdf1,featuresdf1 ,'transform_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('test.csv')
    tdf2 = data_preparation(df2)
    featuresdf2 = tdf2.columns
    data_exporting(tdf2,featuresdf2 ,'transform_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('scorear.csv')
    tdf3 = data_preparation(df3)
    featuresdf3 = tdf3.columns
    data_exporting(tdf3,featuresdf3 ,'transform_score.csv')
    
if __name__ == "__main__":
    main()
