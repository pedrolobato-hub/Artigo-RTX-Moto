#IMPORTANDO BIBLIOTECAS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats as stats
from math import sqrt
from scipy.stats import wilcoxon
import string
from scipy.stats import chi2
import os


#=======================================================================
#1. IMPORTANDO DADOS 
#=======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'diferenca_altimetria_clean.xlsx')
df_alti = pd.read_excel(file_path)

PASTA_GRAFICOS = os.path.join(script_dir, "graficos")
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

multiplicador = 1   # 1 = metros | 100 = centímetros | 1000 = milímetros
print(f"Usando multiplicador: {multiplicador}\n")

#DATAFRAME DADOS ORIGINAIS (DF_ALTI)
col_list = df_alti.columns.tolist()

#DATAFRAME DAS DIFERENÇAS (DF_DIFF)
trats = col_list[2:]
const = 0.138

df_diff = pd.DataFrame(index=df_alti.index)

for trat in trats:
    df_diff[f"{trat}_diff"] = df_alti[trat] - df_alti['NIV'] + const

# Aplicar multiplicador (MÁGICA AQUI)
df_diff = df_diff * multiplicador

#DATAFRAME DAS DIFERENÇAS EM MODULO (DF_DIFF_ABS)
df_diff_abs = df_diff.abs()

print("\ndf_alti.head():\n", df_alti.head())
print("\ndf_diff.head():\n", df_diff.head())
print("\ndf_diff_abs.head():\n", df_diff_abs.head())


### 2. GRAFICO DE PERFIL
for trat in trats:
    df_alti[trat] = df_alti[trat] + const


# --- VARIÁVEIS DE CONFIGURAÇÃO VISUAL ---

# --- Tamanho das bolinhas ---
markersize_ref = 3      # NIV
markersize_sel = 3      # Tratamento 

# --- Espaçamento / margens manual nos eixos ---
x_margin_frac = 0.02    # fração do eixo X
y_margin_frac = 0.05    # fração do eixo Y
# Se quiser alterar, basta mudar esses valores

# --- FUNÇÕES DE ESCALA ---
def escala_percentil(z, p_low=2, p_high=98, margin_frac=0.05):
    """Define limites verticais com base nos percentis e uma margem relativa."""
    y_low = np.percentile(z, p_low)
    y_high = np.percentile(z, p_high)
    y_range = y_high - y_low if (y_high - y_low) != 0 else 1.0
    ymin = y_low - margin_frac * y_range
    ymax = y_high + margin_frac * y_range
    return ymin, ymax

def escala_combinada(arrays, metodo='percentil', margin_frac=y_margin_frac):
    combined = np.concatenate([np.asarray(a).ravel() for a in arrays])
    if metodo == 'percentil':
        ymin, ymax = escala_percentil(combined, margin_frac=margin_frac)
        return ymin, ymax
    else:
        raise ValueError("Método inválido: use 'percentil'")

# --- FUNÇÃO DE PLOT NIV vs UMA COLUNA ---
def plot_niv_vs_trat(df, trat_col, metodo='percentil', nome=None, save=False):
    ref_col = df.columns[1]  # NIV
    y_ref = df[ref_col].to_numpy()
    y_sel = df[trat_col].to_numpy()
    x = np.arange(1, len(y_ref) + 1)

    # calcula limites considerando ambos os conjuntos
    ymin, ymax = escala_combinada([y_ref, y_sel], metodo=metodo, margin_frac=y_margin_frac)

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # --- Plot referência (NIV) ---
    ax.plot(
        x, y_ref,
        marker='o',
        markersize=markersize_ref,   # <- ajuste tamanho da bolinha aqui
        linestyle='-',
        color='black',
        label=f"{ref_col} (referência)",
        alpha=0.9
    )

    # --- Plot seleção (coluna trat) ---
    ax.plot(
        x, y_sel,
        marker='o',
        markersize=markersize_sel,   # <- ajuste tamanho da bolinha aqui
        linestyle='-',
        color='C0',
        label=f"{trat_col}",
        alpha=0.9
    )

    # --- Margens adicionais nos eixos ---
    x_span = len(x)
    ax.set_xlim(1 - x_margin_frac*x_span, len(x) + x_margin_frac*x_span)
    y_span = ymax - ymin
    ax.set_ylim(ymin - y_margin_frac*y_span, ymax + y_margin_frac*y_span)

    # --- Título, rótulos e grid ---
    titulo = f"Perfil Altimétrico — {trat_col}" if nome is None else nome
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel('Número do Ponto', fontsize=14)
    ax.set_ylabel('Altitude (m)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    plt.tight_layout()

    if save:
        fname = f"perfil_NIV_vs_{trat_col}.png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)

# --- Gerar gráficos para cada coluna ---
for trat in trats:
    plot_niv_vs_trat(df_alti, trat, metodo='percentil', save=True)

# --- Gráfico comparativo final: NIV vs todas as colunas ---
def plot_comparativo_todos(df, trats_list, metodo='percentil', save=False):
    ref_col = df.columns[1]
    y_ref = df[ref_col].to_numpy()
    x = np.arange(1, len(y_ref) + 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    # --- NIV ---
    ax.plot(x, y_ref, marker='o', markersize=markersize_ref, linestyle='-', color='black',
            linewidth=1.25, label=f"{ref_col} (referência)", alpha=0.95)

    # --- Todos os tratamentos ---
    colors_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, trat in enumerate(trats_list):
        color = colors_palette[i % len(colors_palette)]
        y = df[trat].to_numpy()
        ax.plot(x, y, marker='o', markersize=markersize_sel, linestyle='-', label=f"{trat}",
                color=color, alpha=0.85)

    # Limites combinados
    arrays = [y_ref] + [df[t].to_numpy() for t in trats_list]
    ymin, ymax = escala_combinada(arrays, metodo=metodo, margin_frac=y_margin_frac)
    x_span = len(x)
    y_span = ymax - ymin
    ax.set_xlim(1 - x_margin_frac*x_span, len(x) + x_margin_frac*x_span)
    ax.set_ylim(ymin - y_margin_frac*y_span, ymax + y_margin_frac*y_span)

    ax.set_title("Perfil Altimétrico — Comparativo: NIV e todos os tratamentos", fontsize=14)
    ax.set_xlabel('Número do Ponto', fontsize=14)
    ax.set_ylabel('Altitude (m)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(ncol=2, fontsize=9, loc='best')

    plt.tight_layout()
    if save:
        fig.savefig(f"{PASTA_GRAFICOS}/GRAFICO_DE_PERFIL.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# Chamada do comparativo
plot_comparativo_todos(df_alti, trats, metodo='percentil', save=True)