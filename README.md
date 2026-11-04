# Insurance-charges
Previsão do valor de seguro de saúde com regressão e machine learning
# Previsão do valor de seguro de saúde (Medical Cost)

Projeto de aprendizado de máquina para prever o valor cobrado pelo seguro (`charges`) a partir da idade, do hábito de fumar e do IMC (`bmi`).

## Etapas

1. Limpeza dos dados (remoção de 1 linha duplicada, conversão de `sex` e `smoker` para 0/1 e de `region` em colunas 0/1)
2. Análise exploratória: boxplots, gráficos de dispersão, normalidade (Q-Q plot e Shapiro) e correlação
3. Regressão linear simples (`age` e `smoker`) e regressão múltipla (`age`, `smoker` e `bmi`), com diagnóstico dos resíduos no `statsmodels`
4. Comparação de 7 algoritmos com validação cruzada: regressão múltipla, SVR, árvore de decisão, Random Forest, XGBoost, LightGBM e CatBoost
5. Teste com dois pacientes fictícios

## Principais achados

- O fato de fumar é o que mais pesa no valor do seguro.
- O IMC só pesa de forma marcante entre os fumantes: a média do seguro dos fumantes com IMC a partir de 30 é cerca do dobro da dos fumantes com IMC menor. Entre não fumantes, quase não muda.
- A regressão linear múltipla não capta essa interação e chega a R² de 0,73 na validação cruzada. Os modelos não lineares ficam entre 0,84 e 0,85.

## Resultado

O melhor modelo foi o LightGBM, com R² de 0,852 na validação cruzada (5 folds) e 0,858 no teste. A diferença para XGBoost, CatBoost, Random Forest e árvore de decisão é pequena e está dentro da variação da validação cruzada.

## Como rodar

```
pip install -r requirements.txt
jupyter lab
```

Abra o notebook `projeto_seguro.ipynb` e execute as células em ordem.

## Dados

Medical Cost Personal Datasets (`insurance.csv`), com 1.338 linhas e 7 colunas, disponível no Kaggle.

## Aviso

Projeto de estudo. Os dados são de um banco de dados de seguros, com idades de 18 a 64 anos, e o modelo não deve ser usado para definir preço de seguro nem para decisões sobre saúde.
