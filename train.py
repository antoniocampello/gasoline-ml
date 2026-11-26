import os
import glob
import gc
import json
import shutil
import boto3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

NOME_BUCKET = os.environ.get('BUCKET_NAME', 'gasoline-regression-model-datalake')
PASTA_LOCAL = './base_combustivel_local'


def main():
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    s3 = boto3.client('s3')

    # 1. Download da Base do S3
    print(f"Buscando arquivos CSV na raiz do bucket '{NOME_BUCKET}'...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=NOME_BUCKET)

    keys_csv = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.csv') and '/' not in obj['Key']:
                    keys_csv.append(obj['Key'])

    print(f'Encontrados {len(keys_csv)} arquivos CSV. Baixando...')
    for key in keys_csv:
        caminho_destino = os.path.join(PASTA_LOCAL, key)
        s3.download_file(NOME_BUCKET, key, caminho_destino)

    arquivos_csv = glob.glob(os.path.join(PASTA_LOCAL, '*.csv'))

    # 2. Leitura e Concatenação
    lista_df = []
    for arquivo in arquivos_csv:
        try:
            temp_df = pd.read_csv(
                arquivo,
                sep=';',
                encoding='latin1',
                low_memory=False,
                on_bad_lines='skip'
            )
            temp_df.columns = temp_df.columns.str.replace('ï»¿', '').str.strip()
            temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
            lista_df.append(temp_df)
        except Exception as e:
            print(f'Erro no arquivo {os.path.basename(arquivo)}: {e}')

    df = pd.concat(lista_df, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    del lista_df
    shutil.rmtree(PASTA_LOCAL, ignore_errors=True)
    gc.collect()

    print(f'Base total unificada! Registros: {len(df):,}')

    # 3. Pré-processamento
    df_clean = df.dropna(subset=['Valor de Venda']).copy()
    del df
    gc.collect()

    colunas_para_remover = [
        'Valor de Compra', 'Complemento', 'CNPJ da Revenda',
        'Nome da Rua', 'Numero Rua', 'Bairro', 'Cep', 'Revenda',
        'Unidade de Medida', 'Municipio'
    ]
    df_clean.drop(columns=colunas_para_remover, errors='ignore', inplace=True)

    if df_clean['Valor de Venda'].dtype == 'object':
        df_clean['Valor de Venda'] = (
            df_clean['Valor de Venda']
            .astype(str)
            .str.replace(',', '.')
            .astype(float)
        )

    if 'Data da Coleta' in df_clean.columns:
        df_clean['Data da Coleta'] = pd.to_datetime(
            df_clean['Data da Coleta'], format='%d/%m/%Y', errors='coerce'
        )
        df_clean['Ano'] = df_clean['Data da Coleta'].dt.year
        df_clean['Mes'] = df_clean['Data da Coleta'].dt.month
        df_clean['Dia'] = df_clean['Data da Coleta'].dt.day
        df_clean['Dia_Semana'] = df_clean['Data da Coleta'].dt.dayofweek  # 0 = Segunda, 6 = Domingo
        df_clean.drop(columns=['Data da Coleta'], inplace=True)

    df_clean.dropna(subset=['Ano', 'Mes', 'Dia', 'Dia_Semana'], inplace=True)
    df_clean['Ano'] = df_clean['Ano'].astype(int)
    df_clean['Mes'] = df_clean['Mes'].astype(int)
    df_clean['Dia'] = df_clean['Dia'].astype(int)
    df_clean['Dia_Semana'] = df_clean['Dia_Semana'].astype(int)

    if 'Bandeira' in df_clean.columns:
        top_bandeiras = df_clean['Bandeira'].value_counts().nlargest(10).index
        df_clean['Bandeira'] = df_clean['Bandeira'].apply(
            lambda x: x if x in top_bandeiras else 'OUTRAS'
        )

    colunas_categoricas = ['Regiao - Sigla', 'Estado - Sigla', 'Produto', 'Bandeira']
    colunas_existentes = [c for c in colunas_categoricas if c in df_clean.columns]

    df_clean = pd.get_dummies(
        df_clean,
        columns=colunas_existentes,
        drop_first=True,
        dtype='float32'
    )

    # 4. Divisão e Treinamento
    y = df_clean['Valor de Venda'].astype(np.float32)
    X = df_clean.drop(columns=['Valor de Venda']).astype(np.float32)
    colunas_modelo = X.columns.tolist()
    num_features = len(colunas_modelo)

    del df_clean
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    del X, y
    gc.collect()

    print("Treinando Regressão Linear (Baseline)...")
    modelo_lr = LinearRegression(n_jobs=-1)
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)

    print("Treinando Random Forest Regressor (Base Total - Otimizado)...")
    modelo_rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=16,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)

    # 5. Métricas para Logs
    def calcular_metricas(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    mae_lr, rmse_lr, r2_lr = calcular_metricas(y_test, y_pred_lr)
    mae_rf, rmse_rf, r2_rf = calcular_metricas(y_test, y_pred_rf)

    df_metricas = pd.DataFrame({
        'Modelo': ['Regressão Linear', 'Random Forest Regressor'],
        'MAE (R$)': [mae_lr, mae_rf],
        'RMSE (R$)': [rmse_lr, rmse_rf],
        'R² Score': [r2_lr, r2_rf]
    })

    print("\n--- TABELA COMPARATIVA DE DESEMPENHO ---")
    print(df_metricas.to_string(index=False))

    # Libera arrays de treino/teste antes de serializar o ONNX
    del X_train, X_test, y_train, y_test, y_pred_lr, y_pred_rf, modelo_lr
    gc.collect()

    # 6. Exportação para Formato ONNX
    print("\nConvertendo o Random Forest para o formato ONNX...")
    tipo_entrada = [('float_input', FloatTensorType([None, num_features]))]
    modelo_onnx = convert_sklearn(modelo_rf, initial_types=tipo_entrada)

    modelo_local = 'modelo_random_forest.onnx'
    with open(modelo_local, 'wb') as f:
        f.write(modelo_onnx.SerializeToString())

    tamanho_mb = os.path.getsize(modelo_local) / (1024 * 1024)
    print(f"Modelo ONNX gerado localmente: {modelo_local} ({tamanho_mb:.1f} MB)")

    # 7. Upload do Modelo ONNX e Metadados para o S3
    print(f"Fazendo upload do modelo ONNX para s3://{NOME_BUCKET}/modelo/{modelo_local}...")
    s3.upload_file(modelo_local, NOME_BUCKET, f'modelo/{modelo_local}')

    metadados = {
        'colunas': colunas_modelo,
        'top_bandeiras': top_bandeiras.tolist()
    }
    arquivo_metadados = 'metadados_modelo.json'
    with open(arquivo_metadados, 'w', encoding='utf-8') as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    print(f"Fazendo upload dos metadados para s3://{NOME_BUCKET}/modelo/{arquivo_metadados}...")
    s3.upload_file(arquivo_metadados, NOME_BUCKET, f'modelo/{arquivo_metadados}')

    print("Treinamento e exportação ONNX concluídos com sucesso!")


if __name__ == '__main__':
    main()