import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Caricamento dataset
df = pd.read_csv("heart.csv")

# Controlla la cartella corrente (dove saranno salvati i grafici)
print("Cartella corrente:", os.getcwd())

# 1. Distribuzione della variabile dipendente
plt.figure(figsize=(6,4))
sns.countplot(x='output', data=df, hue=None,  palette='Set1')
plt.title("Distribuzione della variabile dipendente 'output'")
plt.xlabel("Output (0 = no attacco, 1 = attacco)")
plt.ylabel("Numero di pazienti")
plt.savefig("output_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Distribuzione rispetto all'età
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='age', hue='output', multiple='stack', bins=15, palette='Set1')
plt.title("Distribuzione di 'output' rispetto all'età")
plt.xlabel("Età")
plt.ylabel("Numero di pazienti")
plt.savefig("output_vs_age.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Distribuzione rispetto al sesso
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='output', data=df, palette='Set1')
plt.title("Distribuzione di 'output' rispetto al sesso")
plt.xlabel("Sesso (0 = femmina, 1 = maschio)")
plt.ylabel("Numero di pazienti")
plt.savefig("output_vs_sex.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Distribuzione rispetto al tipo di dolore toracico (cp)
plt.figure(figsize=(8,5))
sns.countplot(x='cp', hue='output', data=df, palette='Set1')
plt.title("Distribuzione di 'output' rispetto al tipo di dolore toracico (cp)")
plt.xlabel("Tipo di dolore toracico")
plt.ylabel("Numero di pazienti")
plt.savefig("output_vs_cp.png", dpi=300, bbox_inches='tight')
plt.show()
