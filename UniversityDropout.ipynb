{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PREDICCION DE LA DESERCION UNIVERSITARIA"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### IMPORTAR LIBRERIAS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split # para dividir el conjunto de datos para el entrenamiento y la prueba\n",
    "from sklearn import metrics # para comprobar la precisión del modelo\n",
    "from sklearn.tree import DecisionTreeClassifier # Árbol de decisión\n",
    "from sklearn.linear_model import LogisticRegression # Regresión Logística\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import roc_curve\n",
    "from sklearn.metrics import roc_auc_score \n",
    "from xgboost import XGBClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### IMPORTANDO LA DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "database_test = pd.read_csv(\"datasets/universitydropout-Test.csv\")\n",
    "database_train = pd.read_csv(\"datasets/universitydropout-Train.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>GESTACION</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>47</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>174</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>626</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>784</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>813</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID  EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES  \\\n",
       "0   47    45       1      6               16.8          0        4   \n",
       "1  174    45       1      7               19.0          0        4   \n",
       "2  626    45       0      7               12.0          1        3   \n",
       "3  784    45       0      8               17.2          0        4   \n",
       "4  813    45       1      8               16.5          0        4   \n",
       "\n",
       "   CONECTIVIDAD  ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR  GESTACION  \\\n",
       "0             2           0                 0                1          0   \n",
       "1             2           0                 1                0          1   \n",
       "2             1           1                 1                1          0   \n",
       "3             1           0                 1                1          0   \n",
       "4             2           1                 1                1          1   \n",
       "\n",
       "   INGRESO_FAMILIAR  \n",
       "0              1200  \n",
       "1              1500  \n",
       "2              1500  \n",
       "3              2200  \n",
       "4              1800  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>GESTACION</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "      <th>DESERCION</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>47</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1200</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>174</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1500</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>626</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1500</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>784</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2200</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>813</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1800</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID  EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES  \\\n",
       "0   47    45       1      6               16.8          0        4   \n",
       "1  174    45       1      7               19.0          0        4   \n",
       "2  626    45       0      7               12.0          1        3   \n",
       "3  784    45       0      8               17.2          0        4   \n",
       "4  813    45       1      8               16.5          0        4   \n",
       "\n",
       "   CONECTIVIDAD  ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR  GESTACION  \\\n",
       "0             2           0                 0                1          0   \n",
       "1             2           0                 1                0          1   \n",
       "2             1           1                 1                1          0   \n",
       "3             1           0                 1                1          0   \n",
       "4             2           1                 1                1          1   \n",
       "\n",
       "   INGRESO_FAMILIAR  DESERCION  \n",
       "0              1200          0  \n",
       "1              1500          1  \n",
       "2              1500          1  \n",
       "3              2200          1  \n",
       "4              1800          0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "database_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EXPLORACION DE LOS DATOS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Verificar la cantidad de datos en los datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1500, 13)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1500, 14)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Verificar el tipo de dato en ambos datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                     int64\n",
       "EDAD                   int64\n",
       "GENERO                 int64\n",
       "CICLO                  int64\n",
       "PROMEDIO_DE_NOTAS    float64\n",
       "DESAPROBO              int64\n",
       "INTERES                int64\n",
       "CONECTIVIDAD           int64\n",
       "ENFERMEDAD             int64\n",
       "FAMILIAR_ENFERMO       int64\n",
       "SOSTEN_FAMILIAR        int64\n",
       "GESTACION              int64\n",
       "INGRESO_FAMILIAR       int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                     int64\n",
       "EDAD                   int64\n",
       "GENERO                 int64\n",
       "CICLO                  int64\n",
       "PROMEDIO_DE_NOTAS    float64\n",
       "DESAPROBO              int64\n",
       "INTERES                int64\n",
       "CONECTIVIDAD           int64\n",
       "ENFERMEDAD             int64\n",
       "FAMILIAR_ENFERMO       int64\n",
       "SOSTEN_FAMILIAR        int64\n",
       "GESTACION              int64\n",
       "INGRESO_FAMILIAR       int64\n",
       "DESERCION              int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Revision de datos nulos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                   0\n",
       "EDAD                 0\n",
       "GENERO               0\n",
       "CICLO                0\n",
       "PROMEDIO_DE_NOTAS    0\n",
       "DESAPROBO            0\n",
       "INTERES              0\n",
       "CONECTIVIDAD         0\n",
       "ENFERMEDAD           0\n",
       "FAMILIAR_ENFERMO     0\n",
       "SOSTEN_FAMILIAR      0\n",
       "GESTACION            0\n",
       "INGRESO_FAMILIAR     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.isnull(database_test).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                   0\n",
       "EDAD                 0\n",
       "GENERO               0\n",
       "CICLO                0\n",
       "PROMEDIO_DE_NOTAS    0\n",
       "DESAPROBO            0\n",
       "INTERES              0\n",
       "CONECTIVIDAD         0\n",
       "ENFERMEDAD           0\n",
       "FAMILIAR_ENFERMO     0\n",
       "SOSTEN_FAMILIAR      0\n",
       "GESTACION            0\n",
       "INGRESO_FAMILIAR     0\n",
       "DESERCION            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.isnull(database_train).sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Revision de las estadisticas de ambos datasets\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>GESTACION</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>750.500000</td>\n",
       "      <td>19.633333</td>\n",
       "      <td>0.498667</td>\n",
       "      <td>2.948667</td>\n",
       "      <td>15.484267</td>\n",
       "      <td>0.268000</td>\n",
       "      <td>3.580000</td>\n",
       "      <td>1.505333</td>\n",
       "      <td>0.521333</td>\n",
       "      <td>0.490667</td>\n",
       "      <td>0.472000</td>\n",
       "      <td>0.495333</td>\n",
       "      <td>1132.373333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>433.157015</td>\n",
       "      <td>7.058997</td>\n",
       "      <td>0.500165</td>\n",
       "      <td>2.201499</td>\n",
       "      <td>2.059173</td>\n",
       "      <td>0.443065</td>\n",
       "      <td>1.082149</td>\n",
       "      <td>0.578095</td>\n",
       "      <td>0.499711</td>\n",
       "      <td>0.500080</td>\n",
       "      <td>0.499382</td>\n",
       "      <td>0.500145</td>\n",
       "      <td>301.400501</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>800.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>375.750000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>13.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>900.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>750.500000</td>\n",
       "      <td>17.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>15.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1125.250000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>17.300000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1300.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1500.000000</td>\n",
       "      <td>45.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2200.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                ID         EDAD       GENERO        CICLO  PROMEDIO_DE_NOTAS  \\\n",
       "count  1500.000000  1500.000000  1500.000000  1500.000000        1500.000000   \n",
       "mean    750.500000    19.633333     0.498667     2.948667          15.484267   \n",
       "std     433.157015     7.058997     0.500165     2.201499           2.059173   \n",
       "min       1.000000    16.000000     0.000000     1.000000          12.000000   \n",
       "25%     375.750000    16.000000     0.000000     1.000000          13.700000   \n",
       "50%     750.500000    17.000000     0.000000     2.000000          15.500000   \n",
       "75%    1125.250000    18.000000     1.000000     4.000000          17.300000   \n",
       "max    1500.000000    45.000000     1.000000    10.000000          19.000000   \n",
       "\n",
       "         DESAPROBO      INTERES  CONECTIVIDAD   ENFERMEDAD  FAMILIAR_ENFERMO  \\\n",
       "count  1500.000000  1500.000000   1500.000000  1500.000000       1500.000000   \n",
       "mean      0.268000     3.580000      1.505333     0.521333          0.490667   \n",
       "std       0.443065     1.082149      0.578095     0.499711          0.500080   \n",
       "min       0.000000     1.000000      1.000000     0.000000          0.000000   \n",
       "25%       0.000000     3.000000      1.000000     0.000000          0.000000   \n",
       "50%       0.000000     4.000000      1.000000     1.000000          0.000000   \n",
       "75%       1.000000     4.000000      2.000000     1.000000          1.000000   \n",
       "max       1.000000     5.000000      3.000000     1.000000          1.000000   \n",
       "\n",
       "       SOSTEN_FAMILIAR    GESTACION  INGRESO_FAMILIAR  \n",
       "count      1500.000000  1500.000000       1500.000000  \n",
       "mean          0.472000     0.495333       1132.373333  \n",
       "std           0.499382     0.500145        301.400501  \n",
       "min           0.000000     0.000000        800.000000  \n",
       "25%           0.000000     0.000000        900.000000  \n",
       "50%           0.000000     0.000000       1000.000000  \n",
       "75%           1.000000     1.000000       1300.000000  \n",
       "max           1.000000     1.000000       2200.000000  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>GESTACION</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "      <th>DESERCION</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>750.500000</td>\n",
       "      <td>19.633333</td>\n",
       "      <td>0.498667</td>\n",
       "      <td>2.948667</td>\n",
       "      <td>15.484267</td>\n",
       "      <td>0.268000</td>\n",
       "      <td>3.580000</td>\n",
       "      <td>1.505333</td>\n",
       "      <td>0.521333</td>\n",
       "      <td>0.490667</td>\n",
       "      <td>0.472000</td>\n",
       "      <td>0.495333</td>\n",
       "      <td>1132.373333</td>\n",
       "      <td>0.401333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>433.157015</td>\n",
       "      <td>7.058997</td>\n",
       "      <td>0.500165</td>\n",
       "      <td>2.201499</td>\n",
       "      <td>2.059173</td>\n",
       "      <td>0.443065</td>\n",
       "      <td>1.082149</td>\n",
       "      <td>0.578095</td>\n",
       "      <td>0.499711</td>\n",
       "      <td>0.500080</td>\n",
       "      <td>0.499382</td>\n",
       "      <td>0.500145</td>\n",
       "      <td>301.400501</td>\n",
       "      <td>0.490332</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>800.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>375.750000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>13.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>900.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>750.500000</td>\n",
       "      <td>17.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>15.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1125.250000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>17.300000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1300.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1500.000000</td>\n",
       "      <td>45.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2200.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                ID         EDAD       GENERO        CICLO  PROMEDIO_DE_NOTAS  \\\n",
       "count  1500.000000  1500.000000  1500.000000  1500.000000        1500.000000   \n",
       "mean    750.500000    19.633333     0.498667     2.948667          15.484267   \n",
       "std     433.157015     7.058997     0.500165     2.201499           2.059173   \n",
       "min       1.000000    16.000000     0.000000     1.000000          12.000000   \n",
       "25%     375.750000    16.000000     0.000000     1.000000          13.700000   \n",
       "50%     750.500000    17.000000     0.000000     2.000000          15.500000   \n",
       "75%    1125.250000    18.000000     1.000000     4.000000          17.300000   \n",
       "max    1500.000000    45.000000     1.000000    10.000000          19.000000   \n",
       "\n",
       "         DESAPROBO      INTERES  CONECTIVIDAD   ENFERMEDAD  FAMILIAR_ENFERMO  \\\n",
       "count  1500.000000  1500.000000   1500.000000  1500.000000       1500.000000   \n",
       "mean      0.268000     3.580000      1.505333     0.521333          0.490667   \n",
       "std       0.443065     1.082149      0.578095     0.499711          0.500080   \n",
       "min       0.000000     1.000000      1.000000     0.000000          0.000000   \n",
       "25%       0.000000     3.000000      1.000000     0.000000          0.000000   \n",
       "50%       0.000000     4.000000      1.000000     1.000000          0.000000   \n",
       "75%       1.000000     4.000000      2.000000     1.000000          1.000000   \n",
       "max       1.000000     5.000000      3.000000     1.000000          1.000000   \n",
       "\n",
       "       SOSTEN_FAMILIAR    GESTACION  INGRESO_FAMILIAR    DESERCION  \n",
       "count      1500.000000  1500.000000       1500.000000  1500.000000  \n",
       "mean          0.472000     0.495333       1132.373333     0.401333  \n",
       "std           0.499382     0.500145        301.400501     0.490332  \n",
       "min           0.000000     0.000000        800.000000     0.000000  \n",
       "25%           0.000000     0.000000        900.000000     0.000000  \n",
       "50%           0.000000     0.000000       1000.000000     0.000000  \n",
       "75%           1.000000     1.000000       1300.000000     1.000000  \n",
       "max           1.000000     1.000000       2200.000000     1.000000  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### PREPROCESAMIENTO DE LA DATA"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Codificar los datos de INGRESO FAMILIAR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Crear varios grupos de ingreso familiar segun nuestros datos\n",
    "#Bandas: 500-1000, 1100-1500, 1600-2000, 2100-2500\n",
    "rangoIngresos = [500, 1000, 1500, 2000, 2500]\n",
    "namesRango= ['0','1', '2', '3']\n",
    "database_test['INGRESO_FAMILIAR'] = pd.cut(database_test['INGRESO_FAMILIAR'], rangoIngresos, labels= namesRango)\n",
    "database_train['INGRESO_FAMILIAR'] = pd.cut(database_train['INGRESO_FAMILIAR'], rangoIngresos, labels= namesRango)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>GESTACION</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>47</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>174</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>626</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>784</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>813</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID  EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES  \\\n",
       "0   47    45       1      6               16.8          0        4   \n",
       "1  174    45       1      7               19.0          0        4   \n",
       "2  626    45       0      7               12.0          1        3   \n",
       "3  784    45       0      8               17.2          0        4   \n",
       "4  813    45       1      8               16.5          0        4   \n",
       "\n",
       "   CONECTIVIDAD  ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR  GESTACION  \\\n",
       "0             2           0                 0                1          0   \n",
       "1             2           0                 1                0          1   \n",
       "2             1           1                 1                1          0   \n",
       "3             1           0                 1                1          0   \n",
       "4             2           1                 1                1          1   \n",
       "\n",
       "  INGRESO_FAMILIAR  \n",
       "0                1  \n",
       "1                1  \n",
       "2                1  \n",
       "3                3  \n",
       "4                2  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>GESTACION</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "      <th>DESERCION</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>47</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>174</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>626</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>784</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>813</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID  EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES  \\\n",
       "0   47    45       1      6               16.8          0        4   \n",
       "1  174    45       1      7               19.0          0        4   \n",
       "2  626    45       0      7               12.0          1        3   \n",
       "3  784    45       0      8               17.2          0        4   \n",
       "4  813    45       1      8               16.5          0        4   \n",
       "\n",
       "   CONECTIVIDAD  ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR  GESTACION  \\\n",
       "0             2           0                 0                1          0   \n",
       "1             2           0                 1                0          1   \n",
       "2             1           1                 1                1          0   \n",
       "3             1           0                 1                1          0   \n",
       "4             2           1                 1                1          1   \n",
       "\n",
       "  INGRESO_FAMILIAR  DESERCION  \n",
       "0                1          0  \n",
       "1                1          1  \n",
       "2                1          1  \n",
       "3                3          1  \n",
       "4                2          0  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Eliminando la columna GESTACION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "database_test.drop(['GESTACION'], axis=1, inplace=True)\n",
    "database_train.drop(['GESTACION'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>47</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>174</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>626</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>784</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>813</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID  EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES  \\\n",
       "0   47    45       1      6               16.8          0        4   \n",
       "1  174    45       1      7               19.0          0        4   \n",
       "2  626    45       0      7               12.0          1        3   \n",
       "3  784    45       0      8               17.2          0        4   \n",
       "4  813    45       1      8               16.5          0        4   \n",
       "\n",
       "   CONECTIVIDAD  ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR  \\\n",
       "0             2           0                 0                1   \n",
       "1             2           0                 1                0   \n",
       "2             1           1                 1                1   \n",
       "3             1           0                 1                1   \n",
       "4             2           1                 1                1   \n",
       "\n",
       "  INGRESO_FAMILIAR  \n",
       "0                1  \n",
       "1                1  \n",
       "2                1  \n",
       "3                3  \n",
       "4                2  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "      <th>DESERCION</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>47</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>174</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>626</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>784</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>813</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID  EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES  \\\n",
       "0   47    45       1      6               16.8          0        4   \n",
       "1  174    45       1      7               19.0          0        4   \n",
       "2  626    45       0      7               12.0          1        3   \n",
       "3  784    45       0      8               17.2          0        4   \n",
       "4  813    45       1      8               16.5          0        4   \n",
       "\n",
       "   CONECTIVIDAD  ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR  \\\n",
       "0             2           0                 0                1   \n",
       "1             2           0                 1                0   \n",
       "2             1           1                 1                1   \n",
       "3             1           0                 1                1   \n",
       "4             2           1                 1                1   \n",
       "\n",
       "  INGRESO_FAMILIAR  DESERCION  \n",
       "0                1          0  \n",
       "1                1          1  \n",
       "2                1          1  \n",
       "3                3          1  \n",
       "4                2          0  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Revision de datos nulos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                   0\n",
       "EDAD                 0\n",
       "GENERO               0\n",
       "CICLO                0\n",
       "PROMEDIO_DE_NOTAS    0\n",
       "DESAPROBO            0\n",
       "INTERES              0\n",
       "CONECTIVIDAD         0\n",
       "ENFERMEDAD           0\n",
       "FAMILIAR_ENFERMO     0\n",
       "SOSTEN_FAMILIAR      0\n",
       "INGRESO_FAMILIAR     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.isnull(database_test).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                   0\n",
       "EDAD                 0\n",
       "GENERO               0\n",
       "CICLO                0\n",
       "PROMEDIO_DE_NOTAS    0\n",
       "DESAPROBO            0\n",
       "INTERES              0\n",
       "CONECTIVIDAD         0\n",
       "ENFERMEDAD           0\n",
       "FAMILIAR_ENFERMO     0\n",
       "SOSTEN_FAMILIAR      0\n",
       "INGRESO_FAMILIAR     0\n",
       "DESERCION            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.isnull(database_train).sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Eliminar columna ID en el dataset de entrenamiento"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "database_train.drop(['ID'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "database_test.drop(['ID'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Codificar los datos de CONECTIVIDAD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Crear varios grupos del grado de conectividad segun nuestros datos\n",
    "#Bandas: 0-1, 2-3\n",
    "rangoConectividad = [0, 1, 2, 3]\n",
    "namesRango= ['0','1', '2']\n",
    "database_test['CONECTIVIDAD'] = pd.cut(database_test['CONECTIVIDAD'], rangoConectividad, labels= namesRango)\n",
    "database_train['CONECTIVIDAD'] = pd.cut(database_train['CONECTIVIDAD'], rangoConectividad, labels= namesRango)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Verificar los datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES CONECTIVIDAD  \\\n",
       "0    45       1      6               16.8          0        4            1   \n",
       "1    45       1      7               19.0          0        4            1   \n",
       "2    45       0      7               12.0          1        3            0   \n",
       "3    45       0      8               17.2          0        4            0   \n",
       "4    45       1      8               16.5          0        4            1   \n",
       "\n",
       "   ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR INGRESO_FAMILIAR  \n",
       "0           0                 0                1                1  \n",
       "1           0                 1                0                1  \n",
       "2           1                 1                1                1  \n",
       "3           0                 1                1                3  \n",
       "4           1                 1                1                2  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "      <th>DESERCION</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16.8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>12.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   EDAD  GENERO  CICLO  PROMEDIO_DE_NOTAS  DESAPROBO  INTERES CONECTIVIDAD  \\\n",
       "0    45       1      6               16.8          0        4            1   \n",
       "1    45       1      7               19.0          0        4            1   \n",
       "2    45       0      7               12.0          1        3            0   \n",
       "3    45       0      8               17.2          0        4            0   \n",
       "4    45       1      8               16.5          0        4            1   \n",
       "\n",
       "   ENFERMEDAD  FAMILIAR_ENFERMO  SOSTEN_FAMILIAR INGRESO_FAMILIAR  DESERCION  \n",
       "0           0                 0                1                1          0  \n",
       "1           0                 1                0                1          1  \n",
       "2           1                 1                1                1          1  \n",
       "3           0                 1                1                3          1  \n",
       "4           1                 1                1                2          0  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Revisar el tipo de datos de los campos codificados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "EDAD                    int64\n",
       "GENERO                  int64\n",
       "CICLO                   int64\n",
       "PROMEDIO_DE_NOTAS     float64\n",
       "DESAPROBO               int64\n",
       "INTERES                 int64\n",
       "CONECTIVIDAD         category\n",
       "ENFERMEDAD              int64\n",
       "FAMILIAR_ENFERMO        int64\n",
       "SOSTEN_FAMILIAR         int64\n",
       "INGRESO_FAMILIAR     category\n",
       "dtype: object"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "EDAD                    int64\n",
       "GENERO                  int64\n",
       "CICLO                   int64\n",
       "PROMEDIO_DE_NOTAS     float64\n",
       "DESAPROBO               int64\n",
       "INTERES                 int64\n",
       "CONECTIVIDAD         category\n",
       "ENFERMEDAD              int64\n",
       "FAMILIAR_ENFERMO        int64\n",
       "SOSTEN_FAMILIAR         int64\n",
       "INGRESO_FAMILIAR     category\n",
       "DESERCION               int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Cambiando los datos typo category a int"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "database_test[['CONECTIVIDAD', 'INGRESO_FAMILIAR']] = database_test[['CONECTIVIDAD', 'INGRESO_FAMILIAR']].astype('int64')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "database_train[['CONECTIVIDAD', 'INGRESO_FAMILIAR']] = database_train[['CONECTIVIDAD', 'INGRESO_FAMILIAR']].astype('int64')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Revisar el tipo de datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "EDAD                   int64\n",
       "GENERO                 int64\n",
       "CICLO                  int64\n",
       "PROMEDIO_DE_NOTAS    float64\n",
       "DESAPROBO              int64\n",
       "INTERES                int64\n",
       "CONECTIVIDAD           int64\n",
       "ENFERMEDAD             int64\n",
       "FAMILIAR_ENFERMO       int64\n",
       "SOSTEN_FAMILIAR        int64\n",
       "INGRESO_FAMILIAR       int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "EDAD                   int64\n",
       "GENERO                 int64\n",
       "CICLO                  int64\n",
       "PROMEDIO_DE_NOTAS    float64\n",
       "DESAPROBO              int64\n",
       "INTERES                int64\n",
       "CONECTIVIDAD           int64\n",
       "ENFERMEDAD             int64\n",
       "FAMILIAR_ENFERMO       int64\n",
       "SOSTEN_FAMILIAR        int64\n",
       "INGRESO_FAMILIAR       int64\n",
       "DESERCION              int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='DESERCION', ylabel='count'>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Obteniendo de forma gráfica a través del método countplotlos los valores diferentes de la columna DESERCION y el número de veces que se repiten\n",
    "#sns.countplot(database_train['DESERCION'])\n",
    "sns.countplot(x = 'DESERCION', data = database_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    898\n",
       "1    602\n",
       "Name: DESERCION, dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Obteniendo los valores diferentes de la columna DESERCION y el número de veces que se repiten\n",
    "database_train['DESERCION'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Revision de las estadisticas de ambos datasets luego del Preprocesamiento"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>19.633333</td>\n",
       "      <td>0.498667</td>\n",
       "      <td>2.948667</td>\n",
       "      <td>15.484267</td>\n",
       "      <td>0.268000</td>\n",
       "      <td>3.580000</td>\n",
       "      <td>0.505333</td>\n",
       "      <td>0.521333</td>\n",
       "      <td>0.490667</td>\n",
       "      <td>0.472000</td>\n",
       "      <td>0.590000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>7.058997</td>\n",
       "      <td>0.500165</td>\n",
       "      <td>2.201499</td>\n",
       "      <td>2.059173</td>\n",
       "      <td>0.443065</td>\n",
       "      <td>1.082149</td>\n",
       "      <td>0.578095</td>\n",
       "      <td>0.499711</td>\n",
       "      <td>0.500080</td>\n",
       "      <td>0.499382</td>\n",
       "      <td>0.683773</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>13.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>17.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>15.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>17.300000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>45.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              EDAD       GENERO        CICLO  PROMEDIO_DE_NOTAS    DESAPROBO  \\\n",
       "count  1500.000000  1500.000000  1500.000000        1500.000000  1500.000000   \n",
       "mean     19.633333     0.498667     2.948667          15.484267     0.268000   \n",
       "std       7.058997     0.500165     2.201499           2.059173     0.443065   \n",
       "min      16.000000     0.000000     1.000000          12.000000     0.000000   \n",
       "25%      16.000000     0.000000     1.000000          13.700000     0.000000   \n",
       "50%      17.000000     0.000000     2.000000          15.500000     0.000000   \n",
       "75%      18.000000     1.000000     4.000000          17.300000     1.000000   \n",
       "max      45.000000     1.000000    10.000000          19.000000     1.000000   \n",
       "\n",
       "           INTERES  CONECTIVIDAD   ENFERMEDAD  FAMILIAR_ENFERMO  \\\n",
       "count  1500.000000   1500.000000  1500.000000       1500.000000   \n",
       "mean      3.580000      0.505333     0.521333          0.490667   \n",
       "std       1.082149      0.578095     0.499711          0.500080   \n",
       "min       1.000000      0.000000     0.000000          0.000000   \n",
       "25%       3.000000      0.000000     0.000000          0.000000   \n",
       "50%       4.000000      0.000000     1.000000          0.000000   \n",
       "75%       4.000000      1.000000     1.000000          1.000000   \n",
       "max       5.000000      2.000000     1.000000          1.000000   \n",
       "\n",
       "       SOSTEN_FAMILIAR  INGRESO_FAMILIAR  \n",
       "count      1500.000000       1500.000000  \n",
       "mean          0.472000          0.590000  \n",
       "std           0.499382          0.683773  \n",
       "min           0.000000          0.000000  \n",
       "25%           0.000000          0.000000  \n",
       "50%           0.000000          0.000000  \n",
       "75%           1.000000          1.000000  \n",
       "max           1.000000          3.000000  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EDAD</th>\n",
       "      <th>GENERO</th>\n",
       "      <th>CICLO</th>\n",
       "      <th>PROMEDIO_DE_NOTAS</th>\n",
       "      <th>DESAPROBO</th>\n",
       "      <th>INTERES</th>\n",
       "      <th>CONECTIVIDAD</th>\n",
       "      <th>ENFERMEDAD</th>\n",
       "      <th>FAMILIAR_ENFERMO</th>\n",
       "      <th>SOSTEN_FAMILIAR</th>\n",
       "      <th>INGRESO_FAMILIAR</th>\n",
       "      <th>DESERCION</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "      <td>1500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>19.633333</td>\n",
       "      <td>0.498667</td>\n",
       "      <td>2.948667</td>\n",
       "      <td>15.484267</td>\n",
       "      <td>0.268000</td>\n",
       "      <td>3.580000</td>\n",
       "      <td>0.505333</td>\n",
       "      <td>0.521333</td>\n",
       "      <td>0.490667</td>\n",
       "      <td>0.472000</td>\n",
       "      <td>0.590000</td>\n",
       "      <td>0.401333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>7.058997</td>\n",
       "      <td>0.500165</td>\n",
       "      <td>2.201499</td>\n",
       "      <td>2.059173</td>\n",
       "      <td>0.443065</td>\n",
       "      <td>1.082149</td>\n",
       "      <td>0.578095</td>\n",
       "      <td>0.499711</td>\n",
       "      <td>0.500080</td>\n",
       "      <td>0.499382</td>\n",
       "      <td>0.683773</td>\n",
       "      <td>0.490332</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>13.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>17.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>15.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>17.300000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>45.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              EDAD       GENERO        CICLO  PROMEDIO_DE_NOTAS    DESAPROBO  \\\n",
       "count  1500.000000  1500.000000  1500.000000        1500.000000  1500.000000   \n",
       "mean     19.633333     0.498667     2.948667          15.484267     0.268000   \n",
       "std       7.058997     0.500165     2.201499           2.059173     0.443065   \n",
       "min      16.000000     0.000000     1.000000          12.000000     0.000000   \n",
       "25%      16.000000     0.000000     1.000000          13.700000     0.000000   \n",
       "50%      17.000000     0.000000     2.000000          15.500000     0.000000   \n",
       "75%      18.000000     1.000000     4.000000          17.300000     1.000000   \n",
       "max      45.000000     1.000000    10.000000          19.000000     1.000000   \n",
       "\n",
       "           INTERES  CONECTIVIDAD   ENFERMEDAD  FAMILIAR_ENFERMO  \\\n",
       "count  1500.000000   1500.000000  1500.000000       1500.000000   \n",
       "mean      3.580000      0.505333     0.521333          0.490667   \n",
       "std       1.082149      0.578095     0.499711          0.500080   \n",
       "min       1.000000      0.000000     0.000000          0.000000   \n",
       "25%       3.000000      0.000000     0.000000          0.000000   \n",
       "50%       4.000000      0.000000     1.000000          0.000000   \n",
       "75%       4.000000      1.000000     1.000000          1.000000   \n",
       "max       5.000000      2.000000     1.000000          1.000000   \n",
       "\n",
       "       SOSTEN_FAMILIAR  INGRESO_FAMILIAR    DESERCION  \n",
       "count      1500.000000       1500.000000  1500.000000  \n",
       "mean          0.472000          0.590000     0.401333  \n",
       "std           0.499382          0.683773     0.490332  \n",
       "min           0.000000          0.000000     0.000000  \n",
       "25%           0.000000          0.000000     0.000000  \n",
       "50%           0.000000          0.000000     0.000000  \n",
       "75%           1.000000          1.000000     1.000000  \n",
       "max           1.000000          3.000000     1.000000  "
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "database_train.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### APLICACION DE ALGORITMOS DE MACHINE LEARNING"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Creando las variables predictoras X y nuestra variable objetivo Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = database_train.drop('DESERCION',axis=1)\n",
    "y = database_train['DESERCION']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Dividimos cada variable para el entrenamiento del modelo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.15,random_state=1) #Separa el 15% de la data para generar las predicciones"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Revision del tamaño de datos de X y Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Para el X_train: (1275, 11) y para el y_train (1275,)\n",
      "Para el X_test: (225, 11) y para el y_test (225,)\n"
     ]
    }
   ],
   "source": [
    "print('Para el X_train:', X_train.shape, 'y para el y_train',y_train.shape)\n",
    "print('Para el X_test:',X_test.shape,'y para el y_test',y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Entrenando el modelo con el algoritmo de Regresion Logística"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\ProgramasDataScience\\envs\\UniversityDropout\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,\n",
       "       1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,\n",
       "       0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,\n",
       "       1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,\n",
       "       0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,\n",
       "       0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0,\n",
       "       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,\n",
       "       0, 0, 0, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logreg = LogisticRegression(random_state=1) #creamos la variable logreg para el modelo de regresión logística\n",
    "logreg_fit=logreg.fit(X_train, y_train) #entrenamos el modelo de regresión logística usando los datos de X_train, y_train\n",
    "logreg_pred = logreg_fit.predict(X_test) #generamos las predicciones con X_test usando el modelo de regresión logística \n",
    "                                                #Calculamos la probabilidades de obtener 1 (sí sobrevivió) con el método predict_proba\n",
    "logreg_pred #mostramos las predicciones generadas 0 cuando no sobrevive 1 cuando sí sobrevive"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  Imprimir el accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "El accuracy para mi primero modelo es :0.71555556\n"
     ]
    }
   ],
   "source": [
    "print('El accuracy para mi primero modelo es :{0:.8f}'.format(accuracy_score(y_test,logreg_pred))) #usando la expresión regex .8f para mostrar 8 decimales"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Probabilidades de obtener 1 (sí desertó) con el método predict_proba"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.27997159, 0.71518809, 0.33538967, 0.38902092, 0.62763327,\n",
       "       0.13413276, 0.13649801, 0.35616323, 0.17272632, 0.74634213,\n",
       "       0.13860874, 0.27619173, 0.37241362, 0.36607298, 0.12955072,\n",
       "       0.1640428 , 0.7206701 , 0.63688046, 0.12738149, 0.64666779,\n",
       "       0.41633319, 0.39874897, 0.68064174, 0.39674906, 0.36776045,\n",
       "       0.41659776, 0.1642065 , 0.70950985, 0.39053843, 0.69832584,\n",
       "       0.4650075 , 0.44488789, 0.14019474, 0.74495646, 0.10359758,\n",
       "       0.42756162, 0.35680448, 0.36240071, 0.61016876, 0.15521873,\n",
       "       0.11372756, 0.28434319, 0.34787908, 0.69398948, 0.46142406,\n",
       "       0.65202067, 0.14569896, 0.65371795, 0.31703775, 0.10346584,\n",
       "       0.42208608, 0.39716285, 0.69924603, 0.69557411, 0.12489188,\n",
       "       0.13808755, 0.4018305 , 0.10408293, 0.42353031, 0.73112529,\n",
       "       0.39030814, 0.36566068, 0.3759587 , 0.73151627, 0.31695741,\n",
       "       0.30728451, 0.40980725, 0.1310712 , 0.1376413 , 0.33744441,\n",
       "       0.42838962, 0.17464061, 0.69464205, 0.16677852, 0.37754794,\n",
       "       0.35003109, 0.13117418, 0.39206057, 0.31211777, 0.65818745,\n",
       "       0.37588769, 0.46083486, 0.13124387, 0.45620338, 0.64627204,\n",
       "       0.45167826, 0.14349178, 0.68551636, 0.44950983, 0.12770008,\n",
       "       0.71001626, 0.13631522, 0.32121599, 0.74379986, 0.63802071,\n",
       "       0.38933689, 0.72970827, 0.35999821, 0.32983752, 0.63773649,\n",
       "       0.12554074, 0.1260748 , 0.14541515, 0.70159007, 0.6509626 ,\n",
       "       0.4692702 , 0.35990631, 0.38773028, 0.69993051, 0.17839856,\n",
       "       0.70253271, 0.42047774, 0.70544755, 0.66382752, 0.14021052,\n",
       "       0.13331133, 0.33205873, 0.40021185, 0.35257391, 0.13938027,\n",
       "       0.15951955, 0.65840358, 0.66941481, 0.35581714, 0.76555576,\n",
       "       0.74561902, 0.31685459, 0.16657399, 0.28707685, 0.36826261,\n",
       "       0.39823324, 0.4127908 , 0.15208202, 0.71688201, 0.30552678,\n",
       "       0.40103793, 0.64470549, 0.73486821, 0.68486633, 0.32544562,\n",
       "       0.73276309, 0.69708962, 0.64962587, 0.37041866, 0.3290249 ,\n",
       "       0.10387623, 0.36587618, 0.128043  , 0.35616927, 0.67224775,\n",
       "       0.13418016, 0.38282446, 0.10684596, 0.41862115, 0.120825  ,\n",
       "       0.42753235, 0.11388938, 0.35964449, 0.45246321, 0.35076949,\n",
       "       0.74200313, 0.38619068, 0.40546815, 0.32508253, 0.34708426,\n",
       "       0.09326348, 0.40015179, 0.15468504, 0.73079655, 0.70349625,\n",
       "       0.10707579, 0.3273675 , 0.37267629, 0.15984179, 0.65452436,\n",
       "       0.69390231, 0.31954324, 0.34907298, 0.30442314, 0.72904837,\n",
       "       0.38461771, 0.14023963, 0.39234097, 0.6700957 , 0.42709337,\n",
       "       0.6453859 , 0.35398897, 0.39280034, 0.38726007, 0.72870993,\n",
       "       0.3826615 , 0.16262481, 0.6212982 , 0.41594634, 0.37342581,\n",
       "       0.65142205, 0.73582287, 0.32722416, 0.13851396, 0.18172006,\n",
       "       0.46207105, 0.37316758, 0.41116179, 0.71033346, 0.349224  ,\n",
       "       0.34227591, 0.40497701, 0.13026096, 0.34450316, 0.41293209,\n",
       "       0.46589147, 0.10821059, 0.37960493, 0.11162547, 0.39993586,\n",
       "       0.30643459, 0.40074896, 0.66495276, 0.15995318, 0.33957081,\n",
       "       0.35265327, 0.33721968, 0.14508071, 0.6794678 , 0.43633374])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Calculamos la probabilidades de obtener 1 (sí sobrevivió) con el método predict_proba\n",
    "proba_pred_test = logreg.predict_proba(X_test)[:,1]\n",
    "proba_pred_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Entrenando el modelo con el algoritmo de DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0,\n",
       "       1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,\n",
       "       1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
       "       0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,\n",
       "       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,\n",
       "       1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,\n",
       "       0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,\n",
       "       0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,\n",
       "       0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0,\n",
       "       0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,\n",
       "       0, 0, 0, 0, 1], dtype=int64)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree_clf = DecisionTreeClassifier(random_state=1) #creamos la variable tree_clf para el modelo de árbol de clasificación\n",
    "clf_fit=tree_clf.fit(X_train,y_train) #entrenamos el modelo de árbol de clasificación usando los datos de X_train, y_train\n",
    "tree_y_pred = clf_fit.predict(X_test)#generamos las predicciones con X_test usando el modelo de árbol de clasificación\n",
    "tree_y_pred  #mostramos las predicciones generadas 0 cuando no deserta 1 cuando sí deserta"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Imprimir el accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "El accuracy para mi segundo modelo es :0.61777778\n"
     ]
    }
   ],
   "source": [
    "print('El accuracy para mi segundo modelo es :{0:.8f}'.format(accuracy_score(y_test,tree_y_pred))) #usando la expresión regex .8f para mostrar 8 decimales"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Probabilidades de obtener 1 (sí desertó) con el método predict_proba"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,\n",
       "       0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,\n",
       "       1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0.,\n",
       "       0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,\n",
       "       0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1.,\n",
       "       1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0.,\n",
       "       0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1.,\n",
       "       1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0.,\n",
       "       1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,\n",
       "       0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1.,\n",
       "       0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,\n",
       "       0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,\n",
       "       1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,\n",
       "       0., 0., 0., 1.])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "proba_pred_test = tree_clf.predict_proba(X_test)[:,1]\n",
    "proba_pred_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Entrenando el modelo con el algoritmo de RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,\n",
       "       1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,\n",
       "       0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,\n",
       "       1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,\n",
       "       0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,\n",
       "       0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0,\n",
       "       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,\n",
       "       0, 0, 0, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rnd_clf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1,max_depth=3,random_state=1) #creamos la variable tree_clf para el modelo de random forest\n",
    "rdn_clf_fit=rnd_clf.fit(X_train,y_train) #entrenamos el modelo de random forest usando los datos de X_train, y_train\n",
    "y_pred_rnd = rdn_clf_fit.predict(X_test) #generamos las predicciones con X_test usando el modelo de random forest\n",
    "y_pred_rnd #mostramos las predicciones generadas 0 cuando no deserta 1 cuando sí deserta"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Imprimir el accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "El accuracy para mi tercer modelo es :0.72000000\n"
     ]
    }
   ],
   "source": [
    "print('El accuracy para mi tercer modelo es :{0:.8f}'.format(accuracy_score(y_test,y_pred_rnd))) #usando la expresión regex .8f para mostrar 8 decimales"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Probabilidades de obtener 1 (sí desertó) con el método predict_proba"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.35331209, 0.55257478, 0.36024296, 0.35878284, 0.58359402,\n",
       "       0.25496703, 0.25087302, 0.36753438, 0.25387017, 0.56124297,\n",
       "       0.24565072, 0.38896981, 0.36370208, 0.35842877, 0.24205053,\n",
       "       0.2626156 , 0.54406561, 0.55105049, 0.25189837, 0.54767498,\n",
       "       0.36273006, 0.42732307, 0.59248524, 0.40682237, 0.33621676,\n",
       "       0.37092188, 0.2574689 , 0.57061145, 0.40957639, 0.56880765,\n",
       "       0.44346654, 0.38641089, 0.25116989, 0.59292413, 0.23714981,\n",
       "       0.36661843, 0.36017393, 0.36354635, 0.48819436, 0.25671024,\n",
       "       0.23984726, 0.38301434, 0.35432682, 0.55561771, 0.42188984,\n",
       "       0.5511211 , 0.24771436, 0.54145256, 0.38080374, 0.23032297,\n",
       "       0.36456978, 0.36447561, 0.59313813, 0.55674562, 0.30207024,\n",
       "       0.24796887, 0.37867128, 0.25414468, 0.40788955, 0.5758785 ,\n",
       "       0.36270289, 0.36667255, 0.3586742 , 0.58792593, 0.37887846,\n",
       "       0.34140494, 0.3930884 , 0.23352481, 0.24917436, 0.36035838,\n",
       "       0.36465724, 0.29980259, 0.55853129, 0.25530752, 0.36132736,\n",
       "       0.40893061, 0.25737017, 0.34586636, 0.36538049, 0.54164784,\n",
       "       0.36733042, 0.44224403, 0.31212903, 0.44086313, 0.55560112,\n",
       "       0.43749   , 0.2482963 , 0.56329283, 0.41995334, 0.24881701,\n",
       "       0.55576096, 0.24977213, 0.38063865, 0.57204999, 0.51958622,\n",
       "       0.36302063, 0.53390638, 0.36427467, 0.35298339, 0.57784412,\n",
       "       0.31508066, 0.23763486, 0.24609817, 0.56553479, 0.53626001,\n",
       "       0.42736062, 0.35854454, 0.34991993, 0.56642128, 0.25703912,\n",
       "       0.58762171, 0.35912593, 0.58997046, 0.55243778, 0.24306952,\n",
       "       0.28721304, 0.35426267, 0.37247176, 0.40499446, 0.24336916,\n",
       "       0.25280119, 0.57927603, 0.55620176, 0.37932413, 0.55825088,\n",
       "       0.60521635, 0.35376569, 0.25705216, 0.35654315, 0.42838865,\n",
       "       0.36504831, 0.36396406, 0.24073492, 0.57078061, 0.34203932,\n",
       "       0.42297703, 0.56079032, 0.56656923, 0.57359107, 0.36102272,\n",
       "       0.59363792, 0.57769328, 0.57980998, 0.36311248, 0.39030714,\n",
       "       0.26114333, 0.36145988, 0.23746576, 0.40186915, 0.53363109,\n",
       "       0.24854249, 0.40618924, 0.2323    , 0.40352986, 0.23947262,\n",
       "       0.37163165, 0.2459152 , 0.37555425, 0.44961516, 0.40094883,\n",
       "       0.57857629, 0.34756833, 0.35116015, 0.35249029, 0.37301507,\n",
       "       0.23595069, 0.35829285, 0.25651791, 0.58427003, 0.56848924,\n",
       "       0.25345112, 0.35778145, 0.40657464, 0.25812595, 0.5803905 ,\n",
       "       0.55794378, 0.35120221, 0.39859911, 0.34279101, 0.55372999,\n",
       "       0.37825435, 0.2571579 , 0.4112954 , 0.54941375, 0.36087288,\n",
       "       0.56377209, 0.35684639, 0.36400388, 0.4034258 , 0.57128629,\n",
       "       0.4045873 , 0.23629404, 0.55454877, 0.36241699, 0.41057901,\n",
       "       0.53571304, 0.58882485, 0.37134604, 0.24666886, 0.259947  ,\n",
       "       0.44175162, 0.31925992, 0.3681977 , 0.59023275, 0.37738077,\n",
       "       0.41469816, 0.42954924, 0.26745958, 0.36359089, 0.35760945,\n",
       "       0.42642213, 0.27344198, 0.40957753, 0.24116058, 0.34972538,\n",
       "       0.37314298, 0.41801997, 0.53707685, 0.2584248 , 0.34658102,\n",
       "       0.3602226 , 0.40287081, 0.25365439, 0.58124738, 0.42753925])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "proba_pred_test = rnd_clf.predict_proba(X_test)[:,1]\n",
    "proba_pred_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(91      1\n",
       " 75      0\n",
       " 1264    1\n",
       " 330     1\n",
       " 1349    1\n",
       "        ..\n",
       " 943     0\n",
       " 1471    1\n",
       " 661     1\n",
       " 1135    1\n",
       " 1171    0\n",
       " Name: DESERCION, Length: 225, dtype: int64,\n",
       " array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,\n",
       "        1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,\n",
       "        0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,\n",
       "        1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,\n",
       "        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,\n",
       "        1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,\n",
       "        0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,\n",
       "        0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,\n",
       "        0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,\n",
       "        0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0,\n",
       "        0, 0, 0, 1, 1]))"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb_model = XGBClassifier(eta=0.1, \n",
    "                                  max_depth=8, \n",
    "                                  colsample_bytree=0.5, \n",
    "                                  scale_pos_weight=1.1, \n",
    "                                  booster='gbtree', \n",
    "                                  use_label_encoder=False,\n",
    "                                  eval_metric='mlogloss')\n",
    "xgb_fit = xgb_model.fit(X_train._get_numeric_data(), np.ravel(y_train, order='C'))\n",
    "xgb_pred = xgb_fit.predict(X_test._get_numeric_data())\n",
    "(y_test, xgb_pred )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "El accuracy para mi cuarto modelo es :0.68000000\n"
     ]
    }
   ],
   "source": [
    "print('El accuracy para mi cuarto modelo es :{0:.8f}'.format(accuracy_score(y_test, xgb_pred ))) #usando la expresión regex .8f para mostrar 8 decimales"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Curva ROC para todos los modelos"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Creamos la variable clasificadores para guardar los modelos actualizados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "clasificadores = [logreg_fit, clf_fit, rdn_clf_fit, xgb_fit] "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Creamos una función para obtener los parámetros de la curva ROC para cada uno de los modelos actualizados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\ProgramasDataScience\\envs\\UniversityDropout\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "tabla_resultados = pd.DataFrame(columns=['clasificadores', 'fpr','tpr','auc'])\n",
    "for cls in clasificadores:\n",
    "    model = cls.fit(X_train, y_train)\n",
    "    yproba = model.predict_proba(X_test)[:,1]\n",
    "    fpr, tpr, _ = roc_curve(y_test, yproba)\n",
    "    auc = roc_auc_score(y_test, yproba)\n",
    "    tabla_resultados = tabla_resultados.append({'clasificadores':None,'fpr':fpr,'tpr':tpr,'auc':auc}, ignore_index=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Mostramos la tabla_resultados declarando como índices el nombre de cada uno de los modelos\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fpr</th>\n",
       "      <th>tpr</th>\n",
       "      <th>auc</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>clasificadores</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>regresion_logistica</th>\n",
       "      <td>[0.0, 0.007751937984496124, 0.0077519379844961...</td>\n",
       "      <td>[0.0, 0.0, 0.020833333333333332, 0.02083333333...</td>\n",
       "      <td>0.726663</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>arbol_clasificacion</th>\n",
       "      <td>[0.0, 0.35658914728682173, 1.0]</td>\n",
       "      <td>[0.0, 0.5833333333333334, 1.0]</td>\n",
       "      <td>0.613372</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>random_forest</th>\n",
       "      <td>[0.0, 0.0, 0.0, 0.007751937984496124, 0.007751...</td>\n",
       "      <td>[0.0, 0.010416666666666666, 0.03125, 0.03125, ...</td>\n",
       "      <td>0.717862</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>XgBoost</th>\n",
       "      <td>[0.0, 0.0, 0.0, 0.03875968992248062, 0.0387596...</td>\n",
       "      <td>[0.0, 0.010416666666666666, 0.0208333333333333...</td>\n",
       "      <td>0.671512</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                   fpr  \\\n",
       "clasificadores                                                           \n",
       "regresion_logistica  [0.0, 0.007751937984496124, 0.0077519379844961...   \n",
       "arbol_clasificacion                    [0.0, 0.35658914728682173, 1.0]   \n",
       "random_forest        [0.0, 0.0, 0.0, 0.007751937984496124, 0.007751...   \n",
       "XgBoost              [0.0, 0.0, 0.0, 0.03875968992248062, 0.0387596...   \n",
       "\n",
       "                                                                   tpr  \\\n",
       "clasificadores                                                           \n",
       "regresion_logistica  [0.0, 0.0, 0.020833333333333332, 0.02083333333...   \n",
       "arbol_clasificacion                     [0.0, 0.5833333333333334, 1.0]   \n",
       "random_forest        [0.0, 0.010416666666666666, 0.03125, 0.03125, ...   \n",
       "XgBoost              [0.0, 0.010416666666666666, 0.0208333333333333...   \n",
       "\n",
       "                          auc  \n",
       "clasificadores                 \n",
       "regresion_logistica  0.726663  \n",
       "arbol_clasificacion  0.613372  \n",
       "random_forest        0.717862  \n",
       "XgBoost              0.671512  "
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tabla_resultados['clasificadores'] = ['regresion_logistica','arbol_clasificacion','random_forest', 'XgBoost']\n",
    "tabla_resultados.set_index('clasificadores', inplace=True)\n",
    "tabla_resultados"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Dibujamos la curva ROC con los parámetros obtenidos para cada uno de los modelos actualizados\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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vPnmm1y4cIHTp0/z9NNPs2TJkjwd57fffuOTTz7h999/59SpU5w4cYL33nsvy23btWuXbsKX5cuXm5MwGPPIP/jgg0yZMiXL/S33vXDhAp6enub57f/66y/efPNNlixZQnR0NNWrV2fQoEF5ei+FgZSUzYMrrhu45rqD4avTj4Y/HH2YOv510m+cmgqXjkBSHPQPNVrrOR58D9yMhMrdoc5ouG0UuMiPR9jepzs+5VB03ucHz4u6/nV5rflruW5XrVo1nnrqKWbPns3hw4e5ceMGX3zxBT///DMXL14kMDCQDz/8kN69ewPGrGKTJ0+mZcuWTJkyhTJlyvDjjz/SvXt3wJj6c9iwYfz999+0bNmSOnXS/79cunQpb7zxBmfPnqVJkyb89NNP5hrc1apVY9SoUcycOZPjx48zcOBAPvroI4YNG8amTZto0aIFCxcuxM/PL8v3kpCQQEBAACkpKTRu3JiKFSty/PhxDh48yFNPPUVYWBhVqlTh448/5v77jfLRw4YNw9PTk/DwcP766y+WLFlC/fr1efbZZ9mwYQM+Pj688MILjB49GoAdO3bw9NNPc+TIETw9PRkyZAhffvkl7du3B6BMmTIArF27llatch8LtHjxYsqUKcPLL7/Mzz//zCuvvJLrPmmuXbvGu+++S2hoKH369DEv79mzJz175q1g1vTp03nsscfMX4jeeecdhgwZwieffGLVvv369cPb27jLqE6dOtSpU4djx47luu+iRYsoX7487dq1A2DZsmX0798/XRxVqlTh+PHjVs1xX1hIiz0PrrnuIF6dybS8jn8detTokX7hHx9A3GXwqw61c5hdSms4/D381gL+eRlSU4wSsZLUxS1i7ty5rFixgqtXr+Lm5kbNmjXZuHEj165d47333uOhhx7i3Llz5u23b99OnTp1iIqK4tVXX+Wxxx4zz0U+ePBg7rzzTqKionjnnXfMs4kBHDlyhEGDBvH1119z6dIlevToQc+ePUlMTDRvs3jxYtauXcuRI0dYtmwZ3bt356OPPiIqKorU1FS+/fbbbN9HyZIliY2NBYw5v48fP05SUhI9e/aka9euXLx4ke+++44hQ4Zw+PBh835z5szhrbfe4vr167Ru3ZqePXvSuHFjzp49y++//87XX3/Nb7/9BsBzzz3Hc889R0xMDMePH+fBB41ev7T54K9evUps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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(8,6))\n",
    "\n",
    "for i in tabla_resultados.index:\n",
    "    plt.plot(tabla_resultados.loc[i]['fpr'], \n",
    "             tabla_resultados.loc[i]['tpr'], \n",
    "             label=\"{}, AUC={:.4f}\".format(i, tabla_resultados.loc[i]['auc']))\n",
    "    \n",
    "plt.plot([0,1], [0,1], color='orange', linestyle='--')\n",
    "\n",
    "plt.xticks(np.arange(0.0, 1.1, step=0.1))\n",
    "plt.xlabel(\"Falsos Positivos\", fontsize=15)\n",
    "\n",
    "plt.yticks(np.arange(0.0, 1.1, step=0.1))\n",
    "plt.ylabel(\"Verdaderos Positivos\", fontsize=15)\n",
    "\n",
    "plt.title('Análisis de la curva ROC', fontweight='bold', fontsize=15)\n",
    "plt.legend(prop={'size':12}, loc='lower right')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Variamos la profundidad de las ramas de los árboles de clasificación"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mi árbol da un accuracy de: 0.6533333333333333 cuando su max_depth es:  1\n",
      "Mi árbol da un accuracy de: 0.7155555555555555 cuando su max_depth es:  2\n",
      "Mi árbol da un accuracy de: 0.7066666666666667 cuando su max_depth es:  3\n",
      "Mi árbol da un accuracy de: 0.7155555555555555 cuando su max_depth es:  4\n",
      "Mi árbol da un accuracy de: 0.72 cuando su max_depth es:  5\n",
      "Mi árbol da un accuracy de: 0.6711111111111111 cuando su max_depth es:  6\n",
      "Mi árbol da un accuracy de: 0.6577777777777778 cuando su max_depth es:  7\n",
      "Mi árbol da un accuracy de: 0.6711111111111111 cuando su max_depth es:  8\n",
      "Mi árbol da un accuracy de: 0.6088888888888889 cuando su max_depth es:  9\n"
     ]
    }
   ],
   "source": [
    "for i in range(1,10):\n",
    "    tree_clf = DecisionTreeClassifier(max_depth=i)\n",
    "    tree_clf.fit(X_train,y_train)\n",
    "    y_pred = tree_clf.predict(X_test)\n",
    "    print(\"Mi árbol da un accuracy de:\", accuracy_score(y_test,y_pred), \"cuando su max_depth es: \", i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mi árbol da un accuracy de: 0.5733333333333334 cuando su max_depth es:  1\n",
      "Mi árbol da un accuracy de: 0.6888888888888889 cuando su max_depth es:  2\n",
      "Mi árbol da un accuracy de: 0.72 cuando su max_depth es:  3\n",
      "Mi árbol da un accuracy de: 0.72 cuando su max_depth es:  4\n",
      "Mi árbol da un accuracy de: 0.7155555555555555 cuando su max_depth es:  5\n",
      "Mi árbol da un accuracy de: 0.72 cuando su max_depth es:  6\n",
      "Mi árbol da un accuracy de: 0.72 cuando su max_depth es:  7\n",
      "Mi árbol da un accuracy de: 0.7111111111111111 cuando su max_depth es:  8\n",
      "Mi árbol da un accuracy de: 0.7022222222222222 cuando su max_depth es:  9\n"
     ]
    }
   ],
   "source": [
    "for i in range(1,10):\n",
    "    rnd_clf = RandomForestClassifier(max_depth=i)\n",
    "    rnd_clf.fit(X_train,y_train)\n",
    "    y_pred = rnd_clf.predict(X_test)\n",
    "    print(\"Mi árbol da un accuracy de:\", accuracy_score(y_test,y_pred), \"cuando su max_depth es: \", i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[16:25:35] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.7155555555555555 cuando su max_depth es:  1\n",
      "[16:25:35] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.6711111111111111 cuando su max_depth es:  2\n",
      "[16:25:35] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.6755555555555556 cuando su max_depth es:  3\n",
      "[16:25:35] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.6488888888888888 cuando su max_depth es:  4\n",
      "[16:25:35] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.64 cuando su max_depth es:  5\n",
      "[16:25:35] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.6311111111111111 cuando su max_depth es:  6\n",
      "[16:25:35] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.6133333333333333 cuando su max_depth es:  7\n",
      "[16:25:36] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.6488888888888888 cuando su max_depth es:  8\n",
      "[16:25:36] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Mi árbol da un accuracy de: 0.6444444444444445 cuando su max_depth es:  9\n"
     ]
    }
   ],
   "source": [
    "for i in range(1,10):\n",
    "    # use_label_encoder=False\n",
    "    xgb_model = XGBClassifier(max_depth=i, use_label_encoder=False)\n",
    "    xgb_model.fit(X_train,y_train)\n",
    "    y_pred = xgb_model.predict(X_test)\n",
    "    print(\"Mi árbol da un accuracy de:\", accuracy_score(y_test,y_pred), \"cuando su max_depth es: \", i)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Actualización de los arboles de clasificacion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "El accuracy para mi tercer modelo es :0.72000000\n"
     ]
    }
   ],
   "source": [
    "tree_clf_N = DecisionTreeClassifier(max_depth=5)\n",
    "tree_clf_2=tree_clf_N.fit(X_train,y_train)\n",
    "predicciones_nuevo_md_DTC = tree_clf_2.predict(X_test)\n",
    "print('El accuracy para mi tercer modelo es :{0:.8f}'.format(accuracy_score(y_test,predicciones_nuevo_md_DTC))) #usando la expresión regex .8f para mostrar 8 decimales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "matriz_confusion = confusion_matrix(y_test,predicciones_nuevo_md_DTC)\n",
    "\n",
    "plt.figure(figsize=(10,6))\n",
    "ax=sns.heatmap(matriz_confusion, annot = True, annot_kws={\"size\": 10}, fmt=\".1f\")\n",
    "ax.set_ylim((0,2))\n",
    "plt.xlabel('Nuestra data dice que')\n",
    "plt.ylabel('Nuestro algoritmo nos dice que')\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "El accuracy para mi tercer modelo es :0.72000000\n"
     ]
    }
   ],
   "source": [
    "rnd_clf_N = RandomForestClassifier(max_depth=3)\n",
    "rnd_clf_2=rnd_clf_N.fit(X_train,y_train)\n",
    "predicciones_nuevo_md_RFC = rnd_clf_2.predict(X_test)\n",
    "print('El accuracy para mi tercer modelo es :{0:.8f}'.format(accuracy_score(y_test,predicciones_nuevo_md_RFC))) #usando la expresión regex .8f para mostrar 8 decimales"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Matriz de confusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 131,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "matriz_confusion = confusion_matrix(y_test,predicciones_nuevo_md_RFC)\n",
    "\n",
    "plt.figure(figsize=(10,6))\n",
    "ax=sns.heatmap(matriz_confusion, annot = True, annot_kws={\"size\": 10}, fmt=\".1f\")\n",
    "ax.set_ylim((0,2))\n",
    "plt.xlabel('Nuestra data dice que')\n",
    "plt.ylabel('Nuestro algoritmo nos dice que')\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[16:27:59] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "El accuracy para mi tercer modelo es :0.71555556\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\ProgramasDataScience\\envs\\UniversityDropout\\lib\\site-packages\\xgboost\\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    }
   ],
   "source": [
    "xgb_model_N = XGBClassifier(max_depth=1)\n",
    "xgb_model_2=xgb_model_N.fit(X_train,y_train)\n",
    "predicciones_nuevo_md_xgb = xgb_model_2.predict(X_test)\n",
    "print('El accuracy para mi tercer modelo es :{0:.8f}'.format(accuracy_score(y_test,predicciones_nuevo_md_xgb))) #usando la expresión regex .8f para mostrar 8 decimales"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Matriz de confusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjIAAAFzCAYAAAAkIOMNAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAkz0lEQVR4nO3deZQdZbWw8Wd3mAMBwpiBeRRQURDhU5TRgUEQEScgYjR6RQQHFFFBcQBUHC/ijXoZBJGoIOAEiARRERnFMCiICBlIgJAEwhVIen9/nEpsku70SZ/U6a5Tz49Vq0+9p07VPr1Wr2z2u9+qyEwkSZKqqGuwA5AkSRooExlJklRZJjKSJKmyTGQkSVJlmchIkqTKMpGRJEmVtdJgB9CXr216pOvCpUHwRJd/etJgOe3Bi6Kd13vusQda+oNfef0t2xpvb4ZsIiNJkkrWvXCwI2iZU0uSJKmyrMhIklRX2T3YEbTMREaSpLrqNpGRJEkVlR1QkbFHRpIkVZYVGUmS6sqpJUmSVFkdMLVkIiNJUl11wH1kTGQkSaqrDqjI2OwrSZIqy4qMJEl1ZbOvJEmqqk64j4yJjCRJdWVFRpIkVVYHVGRs9pUkSZVlRUaSpLryPjKSJKmyOmBqyURGkqS66oBmX3tkJElSZVmRkSSprpxakiRJldUBU0smMpIk1VSmq5YkSVJVdcDUks2+kiSpsqzISJJUV/bISJKkyuqAqSUTGUmS6spHFEiSpMrqgIqMzb6SJKmyrMhIklRXNvtKkqTK6oCpJRMZSZLqqgMqMvbISJKkyrIiI0lSXXVARcZERpKkmvKhkZIkqbqsyEiSpMrqgFVLNvtKkqTKsiIjSVJdObUkSZIqqwOmlkxkJEmqKysykiSpsjqgImOzryRJqiwTGUmS6qq7u7WtHxHxvxExKyKm9BgbGRHXRMR9xc91e7z3iYi4PyL+FhGvbeYrmMhIklRXJScywHnA65YYOwm4NjO3Aa4t9omIHYC3AjsWn/l2RAzr7wImMpIk1VV2t7b1d/rM3wGzlxg+BDi/eH0+cGiP8R9l5jOZ+U/gfmC3/q5hIiNJkgYkIiZExC09tglNfGyjzJwBUPzcsBgfAzzc47ipxdgyuWpJkqS6anH5dWZOBCaumGCI3i7R34dMZCRJqqvBWX49MyJGZeaMiBgFzCrGpwKb9DhuLDC9v5M5tSRJUl2V3+zbmyuAccXrccDlPcbfGhGrRsQWwDbAn/s7mRUZSZLqquSKTERcDOwFrB8RU4FTgTOASRExHngIeDNAZt4VEZOAu4EFwLGZubC/a5jISJKkUmTm2/p4a98+jv8C8IXluYaJjCRJdeWzliRJUmWZyEiSpMrKflc3D3kmMpIk1VUHVGRcfi1JkirLiowkSXXVARUZExlJkupqcO7su0KZyEiSVFcdUJGxR0aSJFWWFRlJkurK5deSJKmyOmBqyURGkqS6MpGRJEmV1QGrlmz2lSRJlWVFRpKkmspum30lSVJV2SMjSZIqqwN6ZExkJEmqqw6YWrLZV5IkVZYVGUmS6soeGUmSVFkmMpIkqbI64FlL9shIkqTKsiKj5fKuP3yN5+b/m+6F3eTChfzwoFNYde3hHPjtDzBi7AbMm/oov3j/t3hm7tNLfXazV7+IvT5zFF3Dupjyo8nc/O0rB+EbSNUVXcH7rvw88x55govGfwWAl497DS8/en+6F3bz99/ewdVnXLzU57Z+9Ys44JSjiGFd3HbJZG44x789FZxaUh39+C1f4N9PPLV4f7djD+bhP9zNzd++kpe9/2Be9v6D+f3plzzvM9EV7PP5cVz6jjN4csZs3n7lafzjmluZfd/0docvVdYex7yOR++fzqprrg7AFnvswPb778LZr/8EC59dwPD1Riz1megKDjrtnZx/5OnMe2Q2773ic9x7zW08ev+0doevocjl1xJsuf8u3P2TGwC4+yc3sNVrdl3qmI133oo5D85k7kOP0v3cQv525Z/Y6jW7tDtUqbJGbDySbffZmVt/dN3isZe9Y19uOOcKFj67AID5j89b6nNjd96K2f+ayRMPP8rC5xby1yv/xPb+7WmR7G5tGwLanshExDHtvqZWoEwOu/Ak3v6Lz/HCt+8NwBrrj2D+rDkAzJ81hzXWX/r/CtfceF2enD578f5TM2az5kbrtiVkqRO8/pSjuOr0i8kezZnrbTmKzXbbngk/+yzvuuRTjH7Rlkt9bq2NRjJ3+uOL9+fNmM0I//a0SHe2tg0Bg1GR+Wxfb0TEhIi4JSJuufGp+9oZk5p0yZtO44cHforLjv4yLz56P8bstl1zH4xYaqgDmuWltth2n5cw//G5zJjy4PPGu4Z1sfqI4Uw89FSu+uIPecvZxy312V7+9J6XDElVV0qPTETc2ddbwEZ9fS4zJwITAb626ZH+pQ1B82fOAeD/Hp/H/VfdysY7b8XTj81j+IbrMH/WHIZvuA5PP7Z0efupGbNZa/TIxftrjhrJ/FlPtCtsqdI23XVbtttvF7bZe2dWWnVlVl1zdd70tf9i3iOzufuqmwGY9pcHyO5kjZFr8fTsJxd/dt4js1l79HqL90eMGsmTRQVVyg5o9i2rIrMRcDRwcC/b48v4nIawlVZflZWHr7b49WZ77sRjf5vKA9fcxg6H7wnADofvyQPX3LrUZx/5ywOsu8XGjNhkA7pWHsZ2B+/OA9fc1tb4par6zZcu4aw9juNrrzyBHx/33/zzj3fz0w+dwz1X38qWe+wAwHpbbMywlVd6XhIDjQRn5OYbs87YDRi28jBeePDu3NvL36hqqgOmlspatfRzYM3MvGPJNyJicknXVMmGbzCCgyeeAEDXSsO492d/5F/X38nMvzzAgeccx45veTVPTn+cn7/vm43jN1qH/c98Nz9751fIhd389tPnc9gPPkYM6+KuS67n8b+7akJqxe2TJnPolyZw7FVnsPC5BVz6ke8AsNaG63DIme/hwmO+TPfCbn5xynkcfcHH6RrWxW2TrufR+/zbU2GINOy2IobqXKlTS9LgeKLLPz1psJz24EW9dDWVZ/7nW/u3dvinLmxrvL1pqiITEasDm2bm30qOR5IktcsQmR5qRb89MhFxMHAH8Otif+eIuKLkuCRJUtm6u1vbhoBmmn0/A+wGzAEo+l42LysgSZLUJjVp9l2QmXOjt5sRSJKk6uqAZt9mEpkpEfF2YFhEbAN8EPhjuWFJkiT1r5mppeOAHYFngIuBecAJJcYkSZLaoQ5TS5n5NPDJYpMkSR2iE+7s228iExHXAUulXZm5TykRSZKk9hgiVZVWNNMj89Eer1cD3gQsKCccSZLUNnVIZDJzyYdy/CEiri8pHkmSpKY1M7U0ssduF7ALsHFpEUmSpPaoyfLrW2n0yASNKaV/AuPLDEqSJLVBTaaWtmhHIJIkqb2yDolMRBy2rPcz89IVF44kSVLzmplaGg/8P+C3xf7ewGRgLo0pJxMZSZKqqA4VGRrJyg6ZOQMgIkYBZ2fmMaVGJkmSylWHG+IBmy9KYgozgW1LikeSJLVLTSoykyPiKhrPWUrgrcB1pUYlSZLKV4dEJjM/EBFvBF5VDE3MzMvKDUuSJKl/zVRkKBIXkxdJkjpIZg0qMpIkqUPVYWpJkiR1qLokMhGxOrBpZv6t5HgkSVKbdMKdfbv6OyAiDgbuAH5d7O8cEVeUHJckSaq4iPhQRNwVEVMi4uKIWC0iRkbENRFxX/Fz3Vau0W8iA3wG2A2YA5CZdwCbt3JRSZI0BHRna9syRMQY4IPArpm5EzCMxi1cTgKuzcxtgGuL/QFrJpFZkJlzW7mIJEkagrpb3Pq3ErB6RKwErAFMBw4Bzi/ePx84tJWv0EyPzJSIeDswLCK2oZFd/bGVi0qSpMFXZo9MZk6LiK8ADwH/B1ydmVdHxEaLnhiQmTMiYsNWrtNMReY4YEfgGeCHNB4WeUIrF5UkSdUXERMi4pYe24Qe761Lo/qyBTAaGB4RR67oGJq5s+/TwCeLTZIkdYoWKzKZORGY2Mfb+wH/zMxHASLiUuD/ATMjYlRRjRkFzGolhmZWLV0TEev02F+3ePaSJEmqsnJ7ZB4Cdo+INSIigH2Be4ArgHHFMeOAy1v5Cs30yKyfmXMW7WTmE63OZ0mSpMFXco/MTRHxE+A2YAFwO43qzZrApIgYTyPZeXMr12kmkemOiE0z8yGAiNiMxlOwJUlSlTW38mjAMvNU4NQlhp+hUZ1ZIZpJZD4J/D4iri/2XwVMWMbxkiRJbdFMs++vI+KlwO5AAB/KzMdKj0ySJJWqEx5R0GciExHbZ+a9RRIDjZvYAGxaTDXdVn54kiSpNCVPLbXDsioyHwHeA5zVy3sJ7FNKRJIkqS2ykxOZzHxP8XPv9oUjSZLappMTmYg4bFkfzMxLV3w4kiRJzVvW1NLBxc8NadyJ77fF/t7AZMBERpKkCuv0qaVjACLi58AOix7wVNxO+Oz2hCdJkkrTyYlMD5svSmIKM4FtS4pHkiS1SUdXZHqYXDxb6WIaq5XeClxXalSSJElNaOaGeB+IiDfSuKMvwMTMvKzcsCRJUtnqUpGhSFxMXiRJ6iC1SWQkSVIHyhjsCFpmIiNJUk11QkWma3kOjoh1I+JFZQUjSZK0PPqtyETEZOANxbF3AI9GxPWZ+eFyQ5MkSWXK7upPLTVTkVk7M+cBhwHnZuYuwH7lhiVJksqW3a1tQ0EzicxKxd18jwB+XnI8kiSpTTKjpW0oaKbZ9zTgKuAPmXlzRGwJ3FduWJIkqWxDparSimZuiPdj4Mc99h8A3lRmUJIkSc3od2opIsZGxGURMSsiZkbETyNibDuCkyRJ5cnuaGkbCprpkTkXuAIYDYwBrizGJElShWW2tg0FzSQyG2TmuZm5oNjOAzYoOS5JklSyulRkHouIIyNiWLEdCTxedmCSJEn9aSaReReNpdePADOAw4sxSZJUYZ1QkWlm1dJDNO7sK0mSOshQ6XNpRTOPKNgAeA+wec/jM9OqjCRJFTZUqiqtaOaGeJcDNwC/ARaWG44kSWqXoXJ33lY0k8iskZkfLz0SSZKk5dRMs+/PI+KA0iORJElt1QkPjWymInM8cHJEPAM8BwSQmTmi1MgkSVKpuuswtZSZa7UjEEmS1F516ZGRJEkdqBNWLTXTIyNJkjQkWZGRJKmmanFDPICIeDGwZ7F7Q2b+pbyQJElSO9RiaikijgcuAjYstgsj4riyA5MkSeXqzmhpGwqaqciMB16emfMBIuJM4EbgW2UGJkmS1J9mEpng+Y8mWFiMSZKkCqvL8utzgZsi4rJi/1Dg+6VFJEmS2qIWzb6Z+dWImAy8kkYl5pjMvL3swCRJUrmGSp9LK5pdfv1PYEFxfETESzPztvLCkiRJZavF1FJEfA54J/APYFERKoF9ygtLkiSpf81UZI4AtsrMZ8sORpIktU8temSAKcA6wKxyQ5EkSe3UCT0ykf2kYxGxK3A5jYTmmUXjmfmGMgN77rEHOiBPlKpn9dF79n+QpFIseHZaWzOLm8e8saV/a1827bJBz4SaqcicD5wJ/BXoLjccSZLULp1QkWkmkXksM79ZeiSSJEnLqZlE5taIOB24gudPLbn8WpKkCuuEHo5mEpmXFD937zHm8mtJkiquLlNL4zPzgZ4DEbFlSfFIkqQ26YQb4nU1ccxPehn78YoORJIkaXn1WZGJiO2BHYG1I+KwHm+NAFYrOzBJklSuTliKvKyppe2Ag2jcDO/gHuNPAu8pMSZJktQGSfWnlvpMZDLzcuDyiNgjM29sY0ySJKkNujtg2dKyppY+lplfAt4eEW9b8v3M/GCpkUmSpFJ1d3JFBrin+HlLOwKRJEmdJyLWAb4H7ETj9i3vAv4GXAJsDjwIHJGZTwzk/MuaWroyIoYBO2XmiQM5uSRJGrra1CPzDeDXmXl4RKwCrAGcDFybmWdExEnAScDHB3LyZS6/zsyFwC4DObEkSRraulvc+hMRI4BXAd8HyMxnM3MOcAiNZzlS/Dx0oN+hmRvi3R4RV9C4d8z8RYOZeelALypJkgZfqxWZiJgATOgxNDEzJ/bY3xJ4FDg3Il4M3AocD2yUmTMAMnNGRGw40BiaSWRGAo/z/EcSJGAiI0lSjRVJy8RlHLIS8FLguMy8KSK+QWMaaYXpN5HJzGNW5AUlSdLQ0IYb4k0FpmbmTcX+T2gkMjMjYlRRjRkFzBroBfp9REFEjI2IyyJiVkTMjIifRsTYgV5QkiQNDWX3yGTmI8DDEbFdMbQvcDdwBTCuGBsHXD7Q79DM1NK5wA+BNxf7RxZj+w/0opIkafC1adXSccBFxYqlB4BjaBRSJkXEeOAh/pNjLLdmEpkNMvPcHvvnRcQJA72gJEkaGrrbkMdk5h3Arr28te+KOH8zT79+LCKOjIhhxXYkjeZfSZKkQdVMIvMu4AjgEWAGcHgxJkmSKqybaGkbCppZtfQQ8IY2xCJJktqoA54Z2X8iExHf7GV4LnBL8YRsSZJUQW1Yfl26ZqaWVgN2Bu4rthfRuEne+Ij4emmRSZKkUnVHtLQNBc2sWtoa2CczFwBExDnA1TSWX/+1xNgkSZKWqZmKzBhgeI/94cDo4oGSz5QSlSRJKl22uA0FzVRkvgTcERGTgaDxFMsvRsRw4DclxiZJkkrUCT0yzaxa+n5E/BLYjUYic3JmTi/ePrHM4CRJUnnacUO8svWZyETES5cYerj4uXFEbJyZt5UXliRJUv+WVZE5axnvJbDPCo5FkiS10VC5qV0r+kxkMnPvdgYiSZLaa6g07LaimWZfImInYAca95QBIDMvKCsoSZJUvo7ukVkkIk4F9qKRyPwSeD3we8BERpKkCuuEVUvN3EfmcBqP2n4kM48BXgysWmpUkiRJTWhmaun/MrM7IhZExAhgFrBlyXFJkqSS1aVH5paIWAf4LnAr8BTw5zKDkiRJ5atFj0xmvr94+Z2I+DUwIjPvLDcsSZJUtk7okWlq1dIimflgSXFIkqQ264REpplmX0mSpCFpuSoykiSpc2QdemQAIuLFwJ7F7g2Z+ZfyQpIkSe1Qi6mliDgeuAjYsNgujIjjyg5MkiSVq7vFbShopiIzHnh5Zs4HiIgzgRuBb5UZmCRJUn+aSWQCWNhjf2ExJkmSKqwuN8T7X+CmiLis2D8U+H5pEUmSpLbo+BviRUQXcBNwPfBKGpWYYzLz9jbEJkmSSjRU+lxascxEpnjG0lmZuQdwW5tikiRJbdAJiUwzN8S7OiLeFBEdUICSJEmdpJkemQ8Dw4EFEfFvGtNLmZkjSo1MkiSVqhbNvpm5VjsCkSRJ7dUJzb7N3BDv2mbGJElStXT0DfEiYjVgDWD9iFiX/9w7ZgQwug2xSZKkEnX61NJ7gRNoJC238p9EZh5wdrlhSZIk9a/PRCYzvwF8IyKOy0wfRyBJUofp7oCaTDPLrx+JiLUAIuJTEXFpRLy05LgkSVLJOqFHpplE5tOZ+WREvBJ4LXA+cE65YUmSpLJli9tQ0Ewis+iBkQcC52Tm5cAq5YUkSZLUnGZuiDctIv4H2A84MyJWpbkESJIkDWFDZXqoFc0kJEcAVwGvy8w5wEjgxDKDkiRJ5euO1rahoJk7+z4dEbNoPP36PmBB8VOSJFVYJ6xa6jeRiYhTgV2B7YBzgZWBC4FXlBuaJEkqU/XTmOamlt4IvAGYD5CZ0wGfvyRJkgZdM82+z2ZmRkQCRMTwkmOSJElt0AnNvs0kMpOKVUvrRMR7gHcB3y03LEmSVLZa9Mhk5lciYn8az1jaDjglM68pPTJJklSq6qcxzVVkKBIXkxdJkjpILaaWIuJJ/pO0rUJj1dL8zBxRZmCSJEn9aWZq6XkrlCLiUGC3sgKSJEnt0Qk9Msv9qIHM/Bmwz4oPRZIktVMnPDSymamlw3rsdtG4Od5QiV+SJA1QLXpkgIN7vF4APAgcUko0kiRJy6GZHplj2hGIJElqr+yACZY+E5mIOGUZn8vM/FwJ8UiSpDbp9Kml+b2MDQfGA+sBJjKSJFVYJ6xa6jORycyzFr2OiLWA44FjgB8BZ/X1OUmSVA3tSGMiYhhwCzAtMw+KiJHAJcDmNPpuj8jMJwZ6/mUuv46IkRHxeeBOGknPSzPz45k5a6AXlCRJtXI8cE+P/ZOAazNzG+DaYn/A+kxkIuLLwM3Ak8ALM/MzrWRMqq5PffGrvOrAt3Loke9bPHbVb2/gkHe8lxe+8gCm3PP3xePTZsxkl70P4U3jjuVN447ls1/6Vq/nnDvvSd59/Mkc8JbxvPv4k5k778nSv4dUZd+deBbTp/6FO26/dqn3Pvyh97Lg2Wmst966vX72ta/Zi7um/I577/49Hzvx2LJDVYV0ky1t/YmIscCBwPd6DB8CnF+8Ph84tJXvsKyKzEeA0cCngOkRMa/YnoyIea1cVNVy6AH7852vfv55Y1tvuRlf/+Kn2WXnnZY6fpMxo/jp+Wfz0/PP5tSPHdfrOb/3g0nsvuvO/PKS77P7rjvz/QsnlRK71CkuuGASBx70jqXGx44dzX77vop//Wtqr5/r6urim9/4AgcdfCQvfPHevOUth/KCF2xTdriqiO4WtyZ8HfjYEodvlJkzAIqfG7byHfpMZDKzKzNXz8y1MnNEj22tZp6zFBHbR8THI+KbEfGN4vULWglWg2PXnV/I2iOe96QKttp8U7bYbOyAz3ndDTdyyOv3A+CQ1+/Hb393Y0sxSp3uht/fxOwn5iw1ftZXPsNJJ3+BzN7/73i3l72Ef/zjQf75z4d47rnnmDTpct5w8GtLjlZVkS3+FxETIuKWHtuEReeOiIOAWZl5a5nfoamnXy+viPg48DYajcF/LobHAhdHxI8y84wyrquhYdqMRzj8ncey5vA1OO4943qt2jz+xBw2WH8kABusP5LZc+a2O0yp8g46aH+mTZvBnXfe3ecxo8dszMNTpy/enzptBru97CXtCE8V0Ory68ycCEzs4+1XAG+IiAOA1YAREXEhMDMiRmXmjIgYBbTUd1tKIkNjifaOmflcz8GI+CpwF9BrIlNkchMAvn3W53n30W8rKTyVZYP11uWaSy9gnbVHcNe99/HBT5zG5Rd+hzWHDx/s0KSOsvrqq3HySR/kdQe8fZnHRcRSY31Vb6QVKTM/AXwCICL2Aj6amUcWPbjjaOQC44DLW7lOWYlMN43+mn8tMT6KZSSAPTO75x57wL+0ClpllVVYZZVVANhx+23YZMwoHnxoGju9YNvnHbfeuuvw6GOz2WD9kTz62GxGrrP2YIQrVdZWW23O5ptvym23XAPA2LGjuPmmq9jjFQcyc+aji4+bNnUGm4wdvXh/7JhRzJgxs+3xamgapDv7ngFMiojxwEPAm1s5WVmJzAnAtRFxH/BwMbYpsDXwgZKuqSFg9hNzWHvEWgwbNoyHp83goYens8mYUUsdt9crd+fyX/2Gdx91BJf/6jfsvecegxCtVF1TptzL6LEvXrx//9//xMv3eD2PP/78xaU333IHW2+9BZtvvgnTpj3CEUccwlFHu3JJDe26s29mTgYmF68fB/ZdUecuJZHJzF9HxLbAbsAYIICpwM2ZubCMa6o8J556Bjfffidz5sxj30OP5P3jj2LtEWty+tfOYfacubz/xFPZfpstmfi1L3DrHVP47+/9gGErDWNYVxennPiBxY3Cp5z+dY449AB2esG2vPuoI/jIp7/IpT+/ilEbbcBXP//JQf6W0tB24Q/O5tWv2oP11x/Jgw/cwmdP+wrnnvejXo8dNWojJn7nyxx8yNEsXLiQ40/4FL/8xQ8Z1tXFeedfwt13/73Xz6l+ujtgmjGG6lypU0vS4Fh99J6DHYJUWwuenbZ0U1OJjtrssJb+rf3Bvy5ta7y9KWtqSZIkDXGdUDEwkZEkqaY6+qGRkiSpsw3SqqUVykRGkqSaateqpTIt8+nXkiRJQ5kVGUmSasoeGUmSVFn2yEiSpMrqhB4ZExlJkmpqqN4Ud3nY7CtJkirLiowkSTVls68kSaose2QkSVJldcKqJXtkJElSZVmRkSSppuyRkSRJldUJy69NZCRJqimbfSVJUmXZ7CtJkjSIrMhIklRTNvtKkqTKstlXkiRVVidUZOyRkSRJlWVFRpKkmuqEVUsmMpIk1VS3PTKSJKmqqp/GmMhIklRbNvtKkiQNIisykiTVVCdUZExkJEmqKW+IJ0mSKsuKjCRJqqxOuI+Mzb6SJKmyrMhIklRT9shIkqTKskdGkiRVVidUZOyRkSRJlWVFRpKkmnJqSZIkVVYnLL82kZEkqaa6O6BHxkRGkqSa6oSKjM2+kiSpsqzISJJUU04tSZKkyuqEqSUTGUmSasqKjCRJqqxOqMjY7CtJkirLiowkSTXl1JIkSaqsTphaMpGRJKmmMrsHO4SW2SMjSZIqy4qMJEk15dOvJUlSZaXNvpIkqao6oSJjj4wkSTWVmS1t/YmITSLiuoi4JyLuiojji/GREXFNRNxX/Fx3oN/BREaSJJVlAfCRzHwBsDtwbETsAJwEXJuZ2wDXFvsD4tSSJEk1VfYN8TJzBjCjeP1kRNwDjAEOAfYqDjsfmAx8fCDXsCIjSVJNZYv/RcSEiLilxzahr2tFxObAS4CbgI2KJGdRsrPhQL+DFRlJkmqq1VVLmTkRmNjfcRGxJvBT4ITMnBcRLV23JxMZSZJqqh2rliJiZRpJzEWZeWkxPDMiRmXmjIgYBcwa6PmdWpIkSaWIRunl+8A9mfnVHm9dAYwrXo8DLh/oNazISJJUU224Id4rgKOAv0bEHcXYycAZwKSIGA88BLx5oBcwkZEkqabasGrp90BfDTH7rohrmMhIklRTnfCIAntkJElSZVmRkSSppjrhWUsmMpIk1VQnTC2ZyEiSVFNlN/u2g4mMJEk1lR0wtWSzryRJqiwrMpIk1ZRTS5IkqbJs9pUkSZXVCT0yJjKSJNVUJ1RkbPaVJEmVZUVGkqSa6oSKjImMJEk1Vf00BqITsjENPRExITMnDnYcUt34t6e6sUdGZZkw2AFINeXfnmrFREaSJFWWiYwkSaosExmVxTl6aXD4t6dasdlXkiRVlhUZSZJUWSYyWqEi4nUR8beIuD8iThrseKS6iIj/jYhZETFlsGOR2slERitMRAwDzgZeD+wAvC0idhjcqKTaOA943WAHIbWbiYxWpN2A+zPzgcx8FvgRcMggxyTVQmb+Dpg92HFI7WYioxVpDPBwj/2pxZgkSaUwkdGKFL2MuSxOklQaExmtSFOBTXrsjwWmD1IskqQaMJHRinQzsE1EbBERqwBvBa4Y5JgkSR3MREYrTGYuAD4AXAXcA0zKzLsGNyqpHiLiYuBGYLuImBoR4wc7JqkdvLOvJEmqLCsykiSpskxkJElSZZnISJKkyjKRkSRJlWUiI0mSKstERhqgiMiIOKvH/kcj4jMr+BonRMQaK/icT/Xz/joR8f4Wr/HOiPjv4vX7IuLoVs4nSX0xkZEG7hngsIhYv8RrnAD0msgUTxsvwzpAS4lMT5n5ncy8YEWdT5J6MpGRBm4BMBH40JJvRMR5EXF4j/2nerw+MSJujog7I+KzxdjwiPhFRPwlIqZExFsi4oPAaOC6iLhu0Xki4rSIuAnYIyJOKc41JSImRsRSz7sq7rR8Y3Hc53qMrxkR10bEbRHx14hY9KTyM4CtIuKOiPjyMo5b8jrHRMTfI+J64BU9xj8TER8tXm8dEb8pvudtEbFVX7+TZZ0/Ir7bo+KzXL9rSZ1lpcEOQKq4s4E7I+JLzRwcEa8BtgF2o/GQzSsi4lXABsD0zDywOG7tzJwbER8G9s7Mx4pTDAemZOYpxXF3Z+ZpxesfAAcBVy5x2W8A52TmBRFxbI/xfwNvzMx5RVXpTxFxBXASsFNm7lycd6Xejssed9OMiFHAZ4FdgLnAdcDtvfwKLgLOyMzLImI1oKuv30lm/m4A5+/3d93zvJKqz4qM1ILMnAdcAHywyY+8pthuB24Dtqfxj+1fgf0i4syI2DMz5/bx+YXAT3vs7x0RN0XEX4F9gB17+cwrgIuL1z/oMR7AFyPiTuA3wBhgo14+38xxLwcmZ+ajmfkscMlSJ4lYCxiTmZcBZOa/M/Np+v6dLNf5e9HMeSVVnBUZqXVfp/EP5bk9xhZQ/I9CMd2zSjEewOmZ+T9LniQidgEOAE6PiKsXVVqW8O/MXFgcvxrwbWDXzHy4aDRerY8Ye3sWyTtoVIJ2ycznIuLBPj7f7HH9Pe9kqWmvHuO9/k6aPP9y/64ldQ4rMlKLMnM2MAno+ZC+B2lMgwAcAqxcvL4KeFdErAkQEWMiYsOIGA08nZkXAl8BXloc/ySwVh+XXpRMPFac7/A+jvsDjSeRQyMpWWRtYFaRnOwNbNbHNfs6rqebgL0iYr2IWBl485IHFNWrqRFxaPHdV43GiqxefyfLcf4HWY7fdS+xS6owKzLSinEWjSd/L/Jd4PKI+DNwLTAfIDOvjogXADcWfblPAUcCWwNfjohu4Dngv4rzTAR+FREzMnPvnhfMzDkR8V0a01IPAjf3EdvxwA8j4niePy11EXBlRNwC3AHcW5z38Yj4Q0RMAX4FnNnbcUvEMqOoCN0IzKBRoeptVdVRwP9ExGnF93zzMn4ns5o8//L+rhefV1L1+fRrSZUTEe+kMaX2gf6OldTZnFqSJEmVZUVGkiRVlhUZSZJUWSYykiSpskxkJElSZZnISJKkyjKRkSRJlWUiI0mSKuv/AwffK2iopkGPAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "matriz_confusion = confusion_matrix(y_test,predicciones_nuevo_md_xgb)\n",
    "\n",
    "plt.figure(figsize=(10,6))\n",
    "ax=sns.heatmap(matriz_confusion, annot = True, annot_kws={\"size\": 10}, fmt=\".1f\")\n",
    "ax.set_ylim((0,2))\n",
    "plt.xlabel('Nuestra data dice que')\n",
    "plt.ylabel('Nuestro algoritmo nos dice que')\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluación de modelos optimizados"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Definimos una lista de métricas y creamos una función para calcular todas las métricas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "lista_metricas=[accuracy_score,precision_score,recall_score,f1_score,roc_auc_score]\n",
    "def calcula_scores_modelo(score_modelo_nlp,real,prediccion):\n",
    "  for metrica in lista_metricas: score_modelo_nlp.append(metrica(real, prediccion))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Calculamos todas las métricas para el modelo de Regresión Logística"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "score_rl=[]\n",
    "calcula_scores_modelo(score_rl,y_test,logreg_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Calculamos todas las métricas para el modelo de DecisionTreeClassifier optimizado"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "score_tcl=[]\n",
    "calcula_scores_modelo(score_tcl,y_test,predicciones_nuevo_md_DTC)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Calculamos todas las métricas para el modelo de RandomForestClassifier optimizado"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "score_rfc=[]\n",
    "calcula_scores_modelo(score_rfc,y_test,predicciones_nuevo_md_RFC)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Calculamos todas las métricas para el modelo de XGBoost optimizado"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "score_xgb=[]\n",
    "calcula_scores_modelo(score_xgb,y_test, predicciones_nuevo_md_xgb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Generamos una tabla donde se consolide todos los indicadores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RegLogistica</th>\n",
       "      <th>DecisionTreeClassifier</th>\n",
       "      <th>RandomForest</th>\n",
       "      <th>XGBoost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>accuracy_score</th>\n",
       "      <td>0.715556</td>\n",
       "      <td>0.720000</td>\n",
       "      <td>0.720000</td>\n",
       "      <td>0.715556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>precision_score</th>\n",
       "      <td>0.766667</td>\n",
       "      <td>0.753846</td>\n",
       "      <td>0.779661</td>\n",
       "      <td>0.766667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>recall_score</th>\n",
       "      <td>0.479167</td>\n",
       "      <td>0.510417</td>\n",
       "      <td>0.479167</td>\n",
       "      <td>0.479167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>f1_score</th>\n",
       "      <td>0.589744</td>\n",
       "      <td>0.608696</td>\n",
       "      <td>0.593548</td>\n",
       "      <td>0.589744</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>roc_auc_score</th>\n",
       "      <td>0.685320</td>\n",
       "      <td>0.693193</td>\n",
       "      <td>0.689196</td>\n",
       "      <td>0.685320</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 RegLogistica  DecisionTreeClassifier  RandomForest   XGBoost\n",
       "accuracy_score       0.715556                0.720000      0.720000  0.715556\n",
       "precision_score      0.766667                0.753846      0.779661  0.766667\n",
       "recall_score         0.479167                0.510417      0.479167  0.479167\n",
       "f1_score             0.589744                0.608696      0.593548  0.589744\n",
       "roc_auc_score        0.685320                0.693193      0.689196  0.685320"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tabla_score = {'RegLogistica': score_rl,\n",
    "               'DecisionTreeClassifier': score_tcl,\n",
    "               'RandomForest': score_rfc,\n",
    "               'XGBoost': score_xgb}\n",
    "tabla_score = pd.DataFrame(tabla_score)  \n",
    "tabla_score.index = ['accuracy_score','precision_score','recall_score','f1_score','roc_auc_score']\n",
    "tabla_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Curva ROC para todos los modelos optimizados"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Creamos la variable clasificadoresOptimizados para guardar los modelos optimizados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [],
   "source": [
    "clasificadoresOptimizados = [logreg_fit, tree_clf_2, rnd_clf_2, xgb_model_2] "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Creamos una función para obtener los parámetros de la curva ROC para cada uno de los modelos optimizados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\ProgramasDataScience\\envs\\UniversityDropout\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[20:14:06] WARNING: ..\\src\\learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\ProgramasDataScience\\envs\\UniversityDropout\\lib\\site-packages\\xgboost\\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    }
   ],
   "source": [
    "tabla_resultados_optimizados = pd.DataFrame(columns=['clasificadores', 'fpr','tpr','auc'])\n",
    "for cls in clasificadoresOptimizados:\n",
    "    model = cls.fit(X_train, y_train)\n",
    "    yproba = model.predict_proba(X_test)[:,1]\n",
    "    fpr, tpr, _ = roc_curve(y_test, yproba)\n",
    "    auc = roc_auc_score(y_test, yproba)\n",
    "    tabla_resultados_optimizados = tabla_resultados_optimizados.append({'clasificadores':None,'fpr':fpr,'tpr':tpr,'auc':auc}, ignore_index=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Mostramos la tabla_resultados_optimizados declarando como índices el nombre de cada uno de los modelos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fpr</th>\n",
       "      <th>tpr</th>\n",
       "      <th>auc</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>clasificadores</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>regresion_logistica</th>\n",
       "      <td>[0.0, 0.007751937984496124, 0.0077519379844961...</td>\n",
       "      <td>[0.0, 0.0, 0.020833333333333332, 0.02083333333...</td>\n",
       "      <td>0.726663</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>arbol_clasificacion</th>\n",
       "      <td>[0.0, 0.0, 0.03875968992248062, 0.046511627906...</td>\n",
       "      <td>[0.0, 0.010416666666666666, 0.15625, 0.1770833...</td>\n",
       "      <td>0.709827</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>random_forest</th>\n",
       "      <td>[0.0, 0.0, 0.0, 0.007751937984496124, 0.007751...</td>\n",
       "      <td>[0.0, 0.010416666666666666, 0.0625, 0.0625, 0....</td>\n",
       "      <td>0.721697</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>XgBoost</th>\n",
       "      <td>[0.0, 0.0, 0.015503875968992248, 0.01550387596...</td>\n",
       "      <td>[0.0, 0.010416666666666666, 0.0104166666666666...</td>\n",
       "      <td>0.729530</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                   fpr  \\\n",
       "clasificadores                                                           \n",
       "regresion_logistica  [0.0, 0.007751937984496124, 0.0077519379844961...   \n",
       "arbol_clasificacion  [0.0, 0.0, 0.03875968992248062, 0.046511627906...   \n",
       "random_forest        [0.0, 0.0, 0.0, 0.007751937984496124, 0.007751...   \n",
       "XgBoost              [0.0, 0.0, 0.015503875968992248, 0.01550387596...   \n",
       "\n",
       "                                                                   tpr  \\\n",
       "clasificadores                                                           \n",
       "regresion_logistica  [0.0, 0.0, 0.020833333333333332, 0.02083333333...   \n",
       "arbol_clasificacion  [0.0, 0.010416666666666666, 0.15625, 0.1770833...   \n",
       "random_forest        [0.0, 0.010416666666666666, 0.0625, 0.0625, 0....   \n",
       "XgBoost              [0.0, 0.010416666666666666, 0.0104166666666666...   \n",
       "\n",
       "                          auc  \n",
       "clasificadores                 \n",
       "regresion_logistica  0.726663  \n",
       "arbol_clasificacion  0.709827  \n",
       "random_forest        0.721697  \n",
       "XgBoost              0.729530  "
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tabla_resultados_optimizados['clasificadores'] = ['regresion_logistica','arbol_clasificacion','random_forest', 'XgBoost']\n",
    "tabla_resultados_optimizados.set_index('clasificadores', inplace=True)\n",
    "tabla_resultados_optimizados"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Dibujamos la curva ROC con los parámetros obtenidos para cada uno de los modelos optimizados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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6pqFeemll/Tw4cNzl48cOaKdnJx0YmJiicdOnDhR9+7du8jtixYt0m3bts1dHj58uH711Vdtimv06NF69OjRxe5T1L9hYIcuJCdWdOO5zsARbZniVSn1IzAICM/ZQWv9j9X+W4CGFRpheTgfCeejAT/mD36SKW8/AEBH+0ZVvrJSwNEN6t8E182AxiPAJF2uKoNJy/YRHpNYrtdoE+DJGwOvsmnf+fPns2LFCvz8/IiNjeXWW29l9uzZ9OvXjz///JM777yTAwcOULduXUaMGEH37t1Zs2YN27ZtY8CAAdx+++15zvf333+zf/9+TCYTixcv5p133mHZsmU0b96cKVOmMHz4cP755x/++OMP1q9fz6FDh/Dy8uLAgQN4e3sXiG/t2rW89NJL/PHHH1x11VU8++yzDBs2jPXr1+fus3z5crZv305iYiKdOnVi4MCBNpfAEhISuPXWW/nss88YPnw4CxYs4NZbb+XIkSP4+vradM8AycnJPP7442zfvp2WLVty6tQpEhISaN26NVOnTmXatGls3Lix0Bi2bdvGvffey8KFC7nppps4depUbtVyvXr1WL58OU2bNmX9+vX079+fa6+9lo4dbfsfbObMmQwcOJChQ4fyxBNPsHz5cu644w6bjj148CBRUVEMGTKkyH02btzIbbcVPULl8uXL6dGjB/v27aNbt26560NCQnB2dubQoUO5pemizJo1q9jHFuvXr+eqqy79vW/ZsoWQkBDatWtHXFwcN910E5999hl16tTJc1xKSgoLFy4sde1DSSq6Kj4QiLJaPmlZV5QHgFXlGlFFOFH4P6ZqSWs49CUsbQpJJ4yGcU3vk6QuivT4448TFBSEq6src+bMYcCAAQwYMACTyUSfPn0IDQ1l5cqVREZGsn37diZPnoyzszM9evQoNMFNnDgRd3d3XF1d+eabb3jppZdo3bo1jo6OvPzyy4SFhREREYGTkxMXL17kwIEDaK1p3bo1/v7+Bc43d+5cxowZQ8eOHXFxceHdd99l8+bNnDhxInefF198EW9vb4KDg7nhhhsICwuz+f5XrFhB8+bNueeee3B0dGT48OG0atWKZcuW2XzPOUwmE3v37iU1NRV/f/88yaY433//PWPGjKFPnz6YTCYCAwNp1aoVALfeeishISEopejVqxd9+/Zlw4YNNp03JSWFBQsWMGLECJycnBgyZEipquPj443eRIX9XnL06NGD8+fPF/nKaVuQlJSEl5dXnmO9vLwKPBvPb8OGDcTGxhb55WL16tXMnDmTyZMn5647efIks2fPZtGiRRw+fJjU1FQee+yxAscuWrQIPz8/evUq2y7QFV1iL6z5sy50R6VuwEjshTYdVUo9BDwExvOQSu3EBvAMAKr5MPvpCbB1DJxcAv79jRK7qHRsLUlXlKCgoNz3ERERLFiwIE8JJjMzkxtuuIGYmBjq1KmDm5tbnmOjoqKKPd8TTzzBM888k7tOa010dDQ33ngjEyZM4NFHHyUyMpLBgwfz4YcfFhi6MyYmJk/p1MPDA19fX6Kjo2ncuDEADRpcmvfBzc2NpKQkm+8/JiaGRo3ydoFt1KgR0dHRNt8zgLu7Oz/99BMffvghDzzwAN27d+ejjz7KTdDFiYqKYsCAAYVuW7VqFZMmTeLQoUOYzWZSUlJo166dTff266+/4ujomHvukSNHcvPNN3P27Fnq1q2Lo6NjgcFXcpadnJzw9TWmtj516hRNmjSx6ZpF8fDwIDExb01VYmIitWsXP7T3zJkzufPOO/Hw8CiwbcuWLYwYMYKFCxfSokWL3PWurq7cf//9uetefvllbr755kLPfe+995b5OBMVXWI/CQRZLTcEYvLvpJS6GpgGDNJaxxd2Iq31t1rrUK11aN26dcsl2DITtRUaXmvvKMrXmQ2wqj3ErISOH0Pv5TKKnLCJ9X9qQUFB3HPPPXlKXMnJybz44ov4+/uTkJBASkpK7v6FJbj85/vmm2/ynC81NTW3Svbxxx9n586d7Nu3j0OHDvHBBx8UOF9AQAARERG5y8nJycTHxxMYWFxlo+3ynx+MVtmBgYE233OOW265hdWrV3Pq1ClatWrF2LFjgZIHOQkKCuLo0aMF1qenp3PnnXfy7LPPEhsby/nz5xkwYEBOe6gSzZw5k6SkJIKDg2nQoAF33XUXmZmZzJ8/HzAKZdY1HwDHjx/HwcGBwMBAWrZsSVBQULEtxgtrZW79yqlduOqqq9i9e3fucceOHSM9PT1PQs4vNTWVBQsWcN999xXYtmvXLm6//XZ++OEHbrrppjzbrr766hI/86ioKNatW8e9995b7H6Xo6IT+3aguVKqiVLKGWNK2Dx9FpRSwcAvwD1a60MVHF/5yEqHhqH2jqJ8HZsBplrQdzO0esoYgEaIUho1ahTLli3j999/Jzs7m7S0NNatW8fJkydp1KgRoaGhTJw4kYyMDDZv3lzis8nx48fz7rvvsm/fPsBozbxgwQIAtm/fztatW8nMzMTd3Z1atWoV2m1sxIgRTJ8+nbCwMNLT03n55Zfp0qVLbmn9Sg0YMIBDhw4xb948srKy+OmnnwgPD+e2224r1T3HxsaydOlSkpOTcXFxwcPDI/d+6tevz8mTJ8nIKLzW8IEHHmD69On8+eefmM1moqOjOXDgABkZGaSnp+eWrletWsUff/yR51ilFOvWrStwzujoaP7880+WL19OWFgYYWFh7N69mxdeeCG3Or5fv34cPHiQ2bNnk5mZSUJCAi+//DJDhgzB0dERpRQff/wxb775JtOnTycxMRGz2czGjRt56KGHgIKtzPO/evbsCRi1BcuWLWPDhg0kJyfz+uuvc8cddxRbYv/111/x9vbmhhtuyLN+79699OvXj88//5yBAwcWOO7+++9n+vTpHDt2jJSUFN57770C7QBmz55Nt27dCAkJKfL6l62wFnXl+QIGAIcwWse/Ylk3HhhveT8NOAeEWV6FtvqzflX6VvHvNNDJmzfq8Jat9AsvT7N3WGUnOUrrCweM95lJWmeU3LpUVLzK3ip+9erVedZt2bJFX3/99drHx0f7+fnpAQMG6IiICK210ZK5R48e2sPDQ99444167NixesyYMVrrgi3Ec8yaNUu3bdtW165dWzds2FDff//9Wmut16xZo9u1a6fd3d21r6+vHjFihL548aLWumCL8a+//lo3bdpU+/j46FtvvVVHRUXlbsPSgjtH/mMLk79V/IYNG3THjh21p6en7tixo96wYUPuNlvvOSYmRl9//fXa09NTe3l56V69eul9+/ZprbVOT0/XAwYM0D4+PtrX17fQOH/55Rfdrl077eHhoUNCQvRvv/2mtdb6iy++0PXq1dNeXl561KhReujQobnHRUVFaQ8PDx0XF1fgHt99913dsWPHAuujo6O1o6Njbg+ETZs26e7du2tvb2/t7++vx4wZoxMSEvIcs2rVKt2jRw/t7u6u/fz8dK9evfTy5cuL/YwLM3fuXB0UFKTd3Nz07bffruPj43O3jRs3To8bNy7P/n379i20dfvo0aO1Ukq7u7vnvtq0aZNnn9dff137+flpPz8/PWrUqAL31LJlSz1tmm35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aeBVpdRJpZRnRcZZnIaZJ+D0f1DLq8R9r0jSMVjTE/Z/ACEPQp0CYxAIIUowceJEm4dJLSu//vorQUFBeHh4sGvXrgq9thBgh37sWuuVWusWWusQrfXblnVTtdZTLe9Pa60baq09tdbelveJxZ+14vRMXQvKAWp5l99FIn6ClR0g8RD0WAidp4KjW/ldTwhRZp599lm++OILkpKSuOaaayrkmidOnEApRVZW6XvJjB49GkdHxzwzu+Wszz/EbGHXmTdvHqGhoXh4eODv70///v3ZuHFjqeP4v//7Pxo0aICXlxdjxowhPb3wQb82bNhQYAwCpRSLFi0CjCFtO3XqhKenJw0bNuT555/PE+8XX3xBaGgoLi4uBcaMzz/GgZubG0opdu7cWer7sScZK74UlM6mR+paaHYTXMakDjZLPQ3e7WBAGATfWeLuQlQHl5OUKqOIiIjLHuwmOzu7jKMpXnJyMosWLcLLy4u5c+eW+viPP/6YJ598kpdffpnY2FgiIyN55JFHWLJkSanO8/vvvzNlyhT+/PNPTpw4wbFjx3jjjTcK3bdnz555xh9Yvnw5Hh4euWO2p6Sk8MknnxAXF8fWrVv5888/+fDDS8OiBAQE8OqrrzJmzJgC584/xsFXX31F06ZN6dixY6nux95kSNkS/BX5F/Fp8QDU0utY554KAc3w+G8TrRIV587+TEZ00VML2uzcbkiNgYD+0PJxaPEomOTXI8ree9ve40BC6ecHL41WdVrxQucXStyvcePGPPzww8ydO5eDBw+SnJzMhx9+yHfffceZM2cICgri7bffZvDgwYAxq9i0adO47rrr+P777/H29uarr76if//+gDH15+jRo/n333+57rrraNmyZZ7rLV26lJdeeono6Gg6dOjA119/nTsGd+PGjXn00UeZPXs2R48eZdiwYbzzzjuMHj2ajRs30qVLFxYsWICPj0+h95Keno6vry/Z2dm0b9+eBg0acPToUfbv38/DDz9MWFgYgYGBvPvuu9x+++2AUSp2dXUlIiKCv//+myVLltCmTRsee+wx1q9fj4eHB0899RSPP/44ANu2beORRx7h0KFDuLq6MnLkSD7++GOuv/56gNwJRVavXk3Xrl1L/PxzJiF59tln+e6773juOdu7z164cIHXX3+d6dOnc8cdd+SuHzhwIAMHDrT5PGCUsh944IHc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l3vy48aN05SUFE/e+/qMk1PVeOzmPOeDZK5lwt+KCSs5TOb83jzRII9zkjeT2acDx/KPcWDfvTSMSKq8I8MIUrY/9xyH162v0XOc1bYNCU8+6Vfbv/71r1x88cUsXLiQF154gcTERE/dQw89REREBNu3bycrK4s+ffqUianuzd69e3n//fe56qqrPGWDBg2iXbt25Obmsn79enr16kXr1q35zW9+w7hx40hPTycjIwMRoX///owdO5YxY8bw4osvcu6553pmEtLT0xER3nzzTRYtWsSMGTPo2bOn3+ORmppKcnIyt9xyC23btmXOnDk88sgjfh1bWFjI0qVLGTNmTIVtcnJyuPTSSyusnzp1KoMHD2bt2rV06NDBUx4REcF5553H2rVrKwxMU8rJYp8DfPvttyfsvqdesUhU9YRIfXfccQclJSVcdtllTJw4sYw249QJ+FS8iPQVkQ0ikikij5dTLyLyslu/WkQCGte0XtZGWu4ETYwjrtcx2t++iojmhTTqlERU//407tcvkHIMo84TExNDu3btKCws5Oabb/aUHzt2jHfffZdnnnmG8PBwLr744nIjbl1++eVER0cTFxdHTk6OJ+jJ5s2bWbx4Mc8//zwNGzakY8eODBkyhDfffBNwtoZ96qmniI+Pp2nTpjz99NOeutDQULZt20Z2djahoaF069btlGKel5KamsrgwYMByoRG9Ye9e/dSUlJS5obHlxYtWrBv374KU+m5CwoKiIqKKnNsVFQU+fn5J9WQk5PDN998U2HEs9WrV/Pss88yceJET1nfvn0ZP348+fn5ZGZm8sYbb1BYWOipnzNnjmcav3v37vTp06fG/SPOFAL6xC4iIcArQC9gC7BCRD5Q1R+9ml0LXOCmzsA092fACAkrJvGmLKLZwHfFXZlWMooZY66t/EDDOA3w90k6ULz11ltkZWXRs2dPRo4cyauvvgrArl27KC4upnnz5p623p9L+f777zn//PM5evQoU6dOpVu3bvz444/k5uYSGxtbJqZ2UlISpZEgc3Nzyzz9JyUlkZubC8Bjjz3G6NGjPQ5dQ4cO5fHHT3gO8YslS5awadMmTxzzwYMHM2rUKDIyMujYsSP169fn6NGjJxx39OhRQkNDiYmJoV69emzbtq3Sp+rKiIyM5MCBA2XKDhw4UGnc8dTUVFJSUmjVqtUJdZmZmVx7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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(8,6))\n",
    "\n",
    "for i in tabla_resultados_optimizados.index:\n",
    "    plt.plot(tabla_resultados_optimizados.loc[i]['fpr'], \n",
    "             tabla_resultados_optimizados.loc[i]['tpr'], \n",
    "             label=\"{}, AUC={:.4f}\".format(i, tabla_resultados_optimizados.loc[i]['auc']))\n",
    "    \n",
    "plt.plot([0,1], [0,1], color='orange', linestyle='--')\n",
    "\n",
    "plt.xticks(np.arange(0.0, 1.1, step=0.1))\n",
    "plt.xlabel(\"Falsos Positivos\", fontsize=15)\n",
    "\n",
    "plt.yticks(np.arange(0.0, 1.1, step=0.1))\n",
    "plt.ylabel(\"Verdaderos Positivos\", fontsize=15)\n",
    "\n",
    "plt.title('Análisis de la curva ROC', fontweight='bold', fontsize=15)\n",
    "plt.legend(prop={'size':12}, loc='lower right')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### PREDICCION UTILIZANDO LOS MODELOS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 1, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_predecida=rnd_clf_2.predict(database_test)\n",
    "test_predecida"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Extraer los Id de la data de origen\n",
    "origen = pd.read_csv('datasets/universitydropout.csv')\n",
    "origen_id=origen['ID']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1500,), (1500,))"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Comprobar que el tamaño de las predicciones y id_origen coincidan para unir ambos dataset sin errores\n",
    "test_predecida.shape, origen_id.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"CELDA N°74\"\n",
    "#Crear un diccionario para guardar ambos dataset y crear un nuevo dataframe llamado PrediccionesDesercion para unir ambos dataset\n",
    "prediccion_dict = {'ID':origen_id,'DESERCION':test_predecida}\n",
    "PrediccionesDesercion = pd.DataFrame(prediccion_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Generar un archivo csv con tu nombre -no olvidar la extensión csv\n",
    "PrediccionesDesercion.to_csv('Prediccion.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "prediccion = pd.read_csv('Prediccion.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>DESERCION</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>47</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>174</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>626</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>784</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>813</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>893</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1011</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>280</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>446</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>465</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>703</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>712</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>839</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>936</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>1083</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>1103</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>1225</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>1259</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>1340</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>1386</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>331</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>366</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>574</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>581</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>689</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>743</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>912</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>1364</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>1428</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>22</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31</th>\n",
       "      <td>192</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>274</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>348</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>351</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>448</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>549</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>616</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>694</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>917</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>1013</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>1047</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>42</th>\n",
       "      <td>1111</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>43</th>\n",
       "      <td>1314</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>44</th>\n",
       "      <td>1390</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>1447</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>46</th>\n",
       "      <td>1448</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>1489</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48</th>\n",
       "      <td>175</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>49</th>\n",
       "      <td>429</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      ID  DESERCION\n",
       "0     47          0\n",
       "1    174          0\n",
       "2    626          1\n",
       "3    784          0\n",
       "4    813          1\n",
       "5    893          0\n",
       "6   1011          0\n",
       "7    280          0\n",
       "8    446          0\n",
       "9    465          0\n",
       "10   703          0\n",
       "11   712          1\n",
       "12   839          0\n",
       "13   936          0\n",
       "14  1083          0\n",
       "15  1103          1\n",
       "16  1225          0\n",
       "17  1259          0\n",
       "18  1340          0\n",
       "19  1386          0\n",
       "20   331          0\n",
       "21   366          0\n",
       "22   574          0\n",
       "23   581          0\n",
       "24   689          0\n",
       "25   743          1\n",
       "26   912          0\n",
       "27  1364          0\n",
       "28  1428          0\n",
       "29    22          0\n",
       "30   172          0\n",
       "31   192          0\n",
       "32   274          0\n",
       "33   348          0\n",
       "34   351          0\n",
       "35   448          1\n",
       "36   549          1\n",
       "37   616          0\n",
       "38   694          0\n",
       "39   917          0\n",
       "40  1013          0\n",
       "41  1047          0\n",
       "42  1111          0\n",
       "43  1314          0\n",
       "44  1390          0\n",
       "45  1447          0\n",
       "46  1448          0\n",
       "47  1489          1\n",
       "48   175          0\n",
       "49   429          0"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediccion.head(50)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
