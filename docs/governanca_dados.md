# Documentação de Governança de Dados – Data Lake

## 1. Introdução

Este documento descreve a estratégia de governança de dados adotada no projeto, incluindo a organização em camadas (Bronze, Silver e Gold), o pipeline de ingestão e o versionamento de dados utilizando o MinIO.

O objetivo é garantir qualidade, rastreabilidade, organização e reprodutibilidade dos dados utilizados no projeto de Inteligência Artificial.

---

## 2. Arquitetura de Dados

A arquitetura segue o modelo Medallion (Bronze, Silver, Gold), amplamente utilizado em pipelines modernos de dados.

### 2.1 Camada Bronze (Dados Brutos)

- Contém os dados originais, sem qualquer tipo de processamento
- Representa a fonte da verdade
- Dados são armazenados exatamente como foram coletados

Características:
- Imutável
- Pode conter inconsistências
- Armazenamento por data

Estrutura:
```sh
/bronze/<fonte>/<data>/raw_data.json
```

---

### 2.2 Camada Silver (Dados Tratados)

- Contém dados limpos e padronizados
- Remove inconsistências e valores nulos
- Define um schema estruturado

Processos realizados:
- Limpeza de dados
- Normalização
- Padronização de colunas

Estrutura:
```sh
/silver/<fonte>/<data>/clean_data.parquet
```

---

### 2.3 Camada Gold (Dados Prontos para Consumo)

- Dados preparados para uso em Machine Learning e análises
- Contém features e transformações finais

Processos realizados:
- Feature engineering
- Preparação para embeddings
- Otimização para leitura

Estrutura:

```sh
/gold/<fonte>/<data>/features.parquet
```

---

## 3. Pipeline de Ingestão de Dados

O pipeline segue o fluxo:
Fonte de Dados → Bronze → Silver → Gold


### 3.1 Etapa 1 – Ingestão (Bronze)
- Coleta de dados via API, arquivos ou scraping
- Armazenamento sem modificações

### 3.2 Etapa 2 – Processamento (Silver)
- Limpeza dos dados
- Padronização de formato
- Conversão para formatos otimizados (Parquet)

### 3.3 Etapa 3 – Enriquecimento (Gold)
- Criação de features
- Preparação para modelos de IA
- Integração com sistemas downstream (ML, APIs, etc.)

---

## 4. Versionamento de Dados (MinIO)

O versionamento é realizado utilizando o MinIO, garantindo histórico e reprodutibilidade.

### 4.1 Estratégia adotada: Versionamento por Data

Os dados são versionados por timestamp, permitindo rastrear todas as execuções do pipeline.


### 4.2 Benefícios
- Facilidade de rollback
- Auditoria de dados
- Comparação entre versões
- Reprodutibilidade de experimentos

---

## 5. Governança e Boas Práticas

### 5.1 Qualidade de Dados
- Validação de schema
- Remoção de dados inconsistentes
- Controle de valores nulos

### 5.2 Rastreabilidade (Data Lineage)
Cada dado possui:
- Origem definida
- Data de ingestão
- Versão associada

### 5.3 Padronização

Estrutura padrão:
```sh
<camada>/<fonte>/<data>/<arquivo>
```

### 5.4 Metadados

Metadados são armazenados para controle do pipeline, incluindo:
- Data de processamento
- Status
- Versão dos dados
- Fonte

---

## 6. Entregável

O sistema final apresenta:

- Estrutura no MinIO com buckets organizados em:
  - Bronze
  - Silver
  - Gold

- Dados organizados nas três camadas:
  - Bruto → Tratado → Enriquecido

- Pipeline funcional:
  - Processo de ingestão e transformação executável

- Versionamento ativo:
  - Histórico organizado por data
  - Possibilidade de recuperação de versões anteriores

---

## 7. Conclusão

A implementação da arquitetura em camadas, juntamente com o versionamento e pipeline de ingestão, garante uma base sólida para aplicações de Inteligência Artificial, promovendo organização, qualidade e governança dos dados ao longo de todo o ciclo de vida.