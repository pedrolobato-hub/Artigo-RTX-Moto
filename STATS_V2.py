#IMPORTANDO BIBLIOTECAS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, chi2, shapiro
from math import sqrt
import string
import os
from pathlib import Path
from scipy import stats

#=======================================================================
# 1.1 CAMINHOS, IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
#=======================================================================

# Diretório onde o script está localizado
script_dir = Path(__file__).parent
pasta_resultados = script_dir / "RESULTS"
pasta_resultados.mkdir(exist_ok=True)

# Carrega o arquivo de dados
file_path = script_dir / 'diferenca_altimetria_clean.xlsx'

df_alti = pd.read_excel(file_path, header=0, sheet_name=None)


# Configurações
multiply = 1  # 1 = metros | 100 = cm | 1000 = mm

for nome_sheet, df_alti in df_alti.items():

    #=======================================================================
    # 1.2 PROCESSAMENTO DOS DATAFRAMES
    #=======================================================================
    # Lista de colunas
    df_dict = pd.read_excel(file_path, sheet_name=None)

    for nome_sheet, df_alti in df_dict.items():
        df_alti.columns = ['PT', 'NIV', 'G1', 'G2', 'G3', 'M1', 'M2', 'M3']

    # Tratamentos (colunas a partir da terceira: V1, V2, V3, ...)
    trats = ['G1','G2','G3','M1','M2','M3']
    const_column = {
        "G1": 1.407,
        "G2": 1.407,
        "G3": 1.407,
        "M1": 1.890,
        "M2": 1.862,
        "M3": 1.880
    }     

    # DataFrame das diferenças

    df_diff = pd.DataFrame(index=df_alti.index)

    for trat in trats:
        const = const_column.get(trat, 0)

        df_diff[f"{trat}_diff"] = (
            (df_alti[trat] - df_alti['NIV']) - const
        ) * multiply

    # Diferenças absolutas
    df_diff_abs = df_diff.abs()

    #=======================================================================
    # 1.3 SALVANDO VISUALIZAÇÕES INICIAIS NO ARQUIVO .TXT
    #=======================================================================
    arquivo_resultados_txt = pasta_resultados / "analise_altimetrica.txt"
    arquivo_saida = open(arquivo_resultados_txt, "w", encoding="utf-8")

    def escrever_no_arquivo(conteudo):
        arquivo_saida.write(str(conteudo) + "\n")
        arquivo_saida.flush()
                    
    escrever_no_arquivo("=== RESULTADOS DA ANÁLISE DE DIFERENÇAS ALTIMÉTRICAS ===")
    escrever_no_arquivo("")

    escrever_no_arquivo("df_alti.head():")
    escrever_no_arquivo(df_alti.head().to_string())

    escrever_no_arquivo("\ndf_diff.head():")
    escrever_no_arquivo(df_diff.head().to_string())

    escrever_no_arquivo("\ndf_diff_abs.head():")
    escrever_no_arquivo(df_diff_abs.head().to_string())

    #=======================================================================
    #2. GRÁFICOS 
    #=======================================================================

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
            fig.savefig(f"{pasta_resultados}/Grafico_de_perfil.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Chamada do comparativo
    plot_comparativo_todos(df_alti, trats, metodo='percentil', save=True)

    #=======================================================================
    ### 3. GRAFICO DE BOXPLOT

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # DISCREPANCIAS
    data_diff = [df_diff[col] for col in df_diff.columns]
    positions_diff = np.arange(1, len(df_diff.columns) + 1)

    # VIOLINGRAPH
    parts1 = ax1.violinplot(data_diff, positions=positions_diff, widths=0.6,
                            showmeans=False, showmedians=False, showextrema=False)
    for pc in parts1['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.5)
        pc.set_edgecolor('none')

    # BOXPLOT
    ax1.boxplot(data_diff, positions=positions_diff,
                notch=True, patch_artist=True,
                boxprops=dict(facecolor='white', color='black'),
                medianprops=dict(color='brown'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))

    ax1.set_title('Discrepâncias por Marcha', fontsize=14)
    ax1.set_ylabel('Discrepância (m)', fontsize=14)
    ax1.set_xlabel('Marcha', fontsize=14)
    ax1.set_xticks(positions_diff)
    ax1.set_xticklabels(df_diff.columns, rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # DISCREPANCIAS ABSOLUTAS
    data_diff_abs = [df_diff_abs[col] for col in df_diff_abs.columns]
    positions_abs = list(range(1, len(df_diff.columns) + 1))

    parts2 = ax2.violinplot(data_diff_abs, positions=positions_abs, widths=0.6,
                            showmeans=False, showmedians=False, showextrema=False)
    for pc in parts2['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.5)
        pc.set_edgecolor('none')

    ax2.boxplot(data_diff_abs, positions=positions_abs,
                notch=True, patch_artist=True,
                boxprops=dict(facecolor='white', color='black'),
                medianprops=dict(color='brown'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))

    ax2.set_title('Discrepâncias Absolutas por Marcha', fontsize=14)
    ax2.set_ylabel('Discrepância Absoluta (m)', fontsize=14)
    ax2.set_xlabel('Marcha', fontsize=14)
    ax2.set_xticks(positions_abs)
    ax2.set_xticklabels(df_diff_abs.columns, rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{pasta_resultados}/Boxplot_com_fundo_violin.png", dpi=300, bbox_inches='tight')

    #========================================================================================
    ### 4. GRÁFICO DE DISTRIBUIÇÃO DE FREQUÊNCIA

    bins = 10

    def CalcHistLin(data, nbins):
        count, bins_values = np.histogram(data, bins=nbins)
        total_count = sum(count)
        percent_value = (count / total_count) * 100
        bins_center = 0.5*(bins_values[:-1] + bins_values[1:])
        return bins_center, percent_value


    # Calcula histogramas
    bins_dict = {}
    freq_dict = {}

    for col in df_diff.columns:
        bin_vals, freq_vals = CalcHistLin(df_diff[col], bins)
        bins_dict[col] = bin_vals
        freq_dict[col] = freq_vals

    # --- Novo: criando ponto inicial correto (xmin_global, 0) ---
    xmin_global = min(b.min() for b in bins_dict.values()) - 0.001

    # Inserir esse ponto no início das curvas
    for col in bins_dict:
        bins_dict[col] = np.insert(bins_dict[col], 0, xmin_global - 0.00001)
        freq_dict[col] = np.insert(freq_dict[col], 0, 0)

    # --- Criar ponto final artificial ---
    def add_final_point(x, y):
        last_x = 2*x[-1] - x[-2]
        x = np.append(x, last_x)
        y = np.append(y, 0)
        return x, y

    for col in bins_dict:
        bins_dict[col], freq_dict[col] = add_final_point(
            bins_dict[col],
            freq_dict[col]
        )

    # --- Plot ---
    plt.figure(figsize=(8,4))
    for col in bins_dict:
        plt.plot(bins_dict[col], freq_dict[col], label=col)

    plt.title("Gráfico de Distribuição de Frequência", fontsize=12)
    plt.xlabel('Erro (m)')
    plt.ylabel('Frequência (%)')

    # Limites automáticos para todas as curvas
    all_freq = np.concatenate(list(freq_dict.values()))
    all_bins = np.concatenate(list(bins_dict.values()))

    plt.ylim(0, all_freq.max() * 1.1)
    plt.xlim(xmin_global, all_bins.max() * 1.1)

    
    plt.grid(True, linestyle=':')
    plt.legend()

    # --- Salvar figura ---
    plt.savefig(f"{pasta_resultados}/Distribuicao_frequencia.png", dpi=300, bbox_inches='tight')

    #===============================================================================

    ### 5. FREQUÊNCIA ACUMULADA POR CLASSES

    bins = 30

    def freq_acumulada_classes(serie, bins):
        # Histograma básico
        counts, bin_edges = np.histogram(serie, bins=bins)
        
        # Frequência percentual por classe
        freq = counts / counts.sum() * 100
        
        # Frequência acumulada
        cum_freq = np.cumsum(freq)
        
        # Centro dos bins
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Inserindo ponto inicial 0%
        bin_centers = np.insert(bin_centers, 0, bin_edges[0])
        cum_freq = np.insert(cum_freq, 0, 0)
        
        return bin_centers, cum_freq

    # Gerar curvas automaticamente para todas as colunas a partir da terceira
    bins_c_dict = {}
    cum_c_dict = {}

    for col in df_diff_abs.columns:
        bins_c_dict[col], cum_c_dict[col] = freq_acumulada_classes(
            df_diff_abs[col], bins
    )

    # Plot
    plt.figure(figsize=(8,4))

    for col in bins_c_dict:
        plt.plot(bins_c_dict[col], cum_c_dict[col], label=col)

    plt.title("Gráfico de Distribuição de Frequência Acumulada", fontsize=12)
    plt.xlabel('Erro Absoluto (m)')
    plt.ylabel('Frequência Acumulada (%)')

    all_bins_c = np.concatenate(list(bins_c_dict.values()))

    plt.xlim(
        all_bins_c.min(),
        all_bins_c.max() * 1.05
    )

    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig(f"{pasta_resultados}/Frequencia_acumulada.png", dpi=300, bbox_inches='tight')

    #=======================================================================

    # 8. GRÁFICO DE MEDIANAS + QUARTIZ

    dados = df_diff.copy()

    # Estatísticas não-paramétricas
    medianas = dados.median()
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1

    plt.figure(figsize=(8,6))
    x = np.arange(len(medianas))

    colors = ["#000000", "#444444", "#888888"]

    plt.errorbar(
        x, medianas,
        yerr=[medianas - q1, q3 - medianas],
        fmt='o',
        markersize=10,
        markeredgewidth=1.2,
        fillstyle='none',
        color='#000000',
        ecolor='#777B7E',
        elinewidth=1.5,
        capsize=8,
        zorder=3
    )

    plt.xticks(x, medianas.index, fontsize=12)
    plt.ylabel("Erro (m)", fontsize=13)
    plt.title("Gráfico de Medianas ± 1° e 3° Quartis", fontsize=15)

    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
    plt.margins(x=0.15, y=0.18)

    ax = plt.gca()
    ax.set_facecolor("#f8f8f8")

    plt.savefig(f"{pasta_resultados}/Medianas_quartil.png", dpi=300, bbox_inches='tight')

    #=======================================================================
    # TESTES COM TABELA
    #=======================================================================

    # 6. TESTE DE NORMALIDADE

    # --- FUNÇÃO ---
    def testar_normalidade_df(df):
        resultados = []
        for col in df.columns:
            dados = df[col].dropna()  # remove valores nulos
            
            if len(dados) < 8:
                resultados.append((col, None, None, "Amostra muito pequena"))
                continue
            
            stat, p = shapiro(dados)
            resultado = "Normal (p > 0.05)" if p > 0.05 else "Não normal (p < 0.05)"
            resultados.append((col, stat, p, resultado))
        
        # Cria um DataFrame com os resultados
        df_result = pd.DataFrame(resultados, columns=["Coluna", "W", "p", "Resultado"])
        return df_result

    df_normalidade = testar_normalidade_df(df_diff)
    escrever_no_arquivo("===== RESULTADOS DO TESTE DE NORMALIDADE (Shapiro–Wilk) =====")
    escrever_no_arquivo(df_normalidade.to_string(index=False))

    # -----------------------------------------------------------------------
    #           TESTES DE COMPARAÇÃO DE MÉDIAS (NÃO PARAMÉTRICOS)
    # -----------------------------------------------------------------------
    # 7. TESTE DE FRIEDMAN + WILCOXON (apenas com os dados com sinal)
    # -----------------------------------------------------------------------
    cols_friedman = df_diff.columns[:]
    dados_friedman = [df_diff[col] for col in cols_friedman]
    stat_friedman, p_friedman = friedmanchisquare(*dados_friedman)

    # Salva no arquivo .txt
    escrever_no_arquivo("\n===== TESTE DE FRIEDMAN =====")
    escrever_no_arquivo(f"Estatística Q = {stat_friedman:.6f}")
    escrever_no_arquivo(f"p-valor = {p_friedman:.6f}")
    if p_friedman < 0.05:
        escrever_no_arquivo("→ Diferença significativa entre os grupos (p < 0.05).")
    else:
        escrever_no_arquivo("→ NÃO há diferença significativa entre os grupos (p ≥ 0.05).")

    # -----------------------------------------------------------------------
    # 2. TESTES DE WILCOXON (só se Friedman for significativo)
    # -----------------------------------------------------------------------
    if p_friedman < 0.05:

        # Colunas a partir da terceira
        colunas = df_diff.columns[:]

        resultados_sinal = []
        resultados_abs = []

        # Combinação manual 2 a 2
        for i in range(len(colunas)):
            for j in range(i + 1, len(colunas)):

                c1 = colunas[i]
                c2 = colunas[j]

                # Extrai nome limpo (ex: V1_diff → V1)
                n1 = c1.replace('_diff', '')
                n2 = c2.replace('_diff', '')

                # --- Sinal ---
                stat_w, p_w = wilcoxon(df_diff[c1], df_diff[c2])
                concl = "Diferença significativa" if p_w < 0.05 else "Sem diferença significativa"

                resultados_sinal.append({
                    "Comparação": f"{n1} × {n2}",
                    "Estatística W": stat_w,
                    "p-valor": p_w,
                    "Conclusão": concl
                })

                # --- Absoluto ---
                stat_w_abs, p_w_abs = wilcoxon(df_diff_abs[c1], df_diff_abs[c2])
                concl_abs = "Diferença significativa" if p_w_abs < 0.05 else "Sem diferença significativa"

                resultados_abs.append({
                    "Comparação (abs)": f"{n1} × {n2}",
                    "W (abs)": stat_w_abs,
                    "p-valor (abs)": p_w_abs,
                    "Conclusão (abs)": concl_abs
                })

        # Salvar resultados do sinal
        df_wilcoxon_sinal = pd.DataFrame(resultados_sinal)
        escrever_no_arquivo("\n===== RESUMO DOS TESTES DE WILCOXON COM SINAL =====")
        escrever_no_arquivo(df_wilcoxon_sinal.to_string(index=False))

        # Salvar resultados do absoluto
        df_wilcoxon_abs = pd.DataFrame(resultados_abs)
        escrever_no_arquivo("\n===== RESUMO DOS TESTES DE WILCOXON ABSOLUTOS =====")
        escrever_no_arquivo(df_wilcoxon_abs.to_string(index=False))


    else:
        # Friedman não significativo → pula Wilcoxon
        msg = "\nPós-testes de Wilcoxon não realizados (Friedman não significativo)."
        print(msg)
        escrever_no_arquivo(msg)


    #=======================================================================
    # 10. TESTE DE NEMENYI
    #=======================================================================

    def nemenyi_pairwise(df):

        k = df.shape[1]    
        n = df.shape[0]     

        rankings = df.rank(axis=1)
        R = rankings.mean()

        # Tabela de valores críticos (α = 0.05)
        q_criticos = {3: 2.343, 4: 2.569, 5: 2.728, 6:2.850, 7:2.949, 8:3.031, 9:3.102, 10:3.164}
        q = q_criticos[k]

        
        CD = q * sqrt(k*(k+1)/(6*n))
        colunas = df.columns.tolist()
        resultados = []

        for i in range(len(colunas)):
            for j in range(i + 1, len(colunas)):
                c1 = colunas[i]
                c2 = colunas[j]

                diff = abs(R[c1] - R[c2])
                sig = "Significativo" if diff > CD else "Não significativo"

                resultados.append([f"{c1} × {c2}", diff, sig])
        

        return CD, resultados
    
    CD_sinal, resultados_sinal = nemenyi_pairwise(df_diff)
    CD_abs, resultados_abs = nemenyi_pairwise(df_diff_abs)


    tabela_sinal = pd.DataFrame(resultados_sinal, 
                                columns=["Comparação (sinal)", "Dif. Ranking (sinal)", "Conclusão (sinal)"])

    tabela_abs = pd.DataFrame(resultados_abs, 
                            columns=["Comparação (abs)", "Dif. Ranking (abs)", "Conclusão (abs)"])

    tabela_final = pd.concat([tabela_sinal, tabela_abs], axis=1)

    escrever_no_arquivo("\n===== TESTE DE NEMENYI — COM SINAL × ABS =====")
    escrever_no_arquivo(tabela_final.to_string(index=False))

    #=======================================================================
    # 11. PEC ALTIMÉTRICO 
    #=======================================================================
    # Tem suas bases em:
    # Especificação Técnica para Controle de Qualidade de Dados Geoespaciais (ET-CQDG).
    # 1) CONFIGURAÇÕES
    escala = 2000  # ajuste conforme seu caso
    alpha = 0.10

    # 1:2.000, 1:5.000 e 1:10.000 são as escalas limite da ET-CQDG
    limites_pec_pcd = {
        2000: {"A": 0.25, "B": 0.50, "C": 0.75, "D": 1.00},
        5000: {"A": 0.50, "B": 1.00, "C": 1.50, "D": 2.00},
        10000: {"A": 1.00, "B": 2.00, "C": 3.00, "D": 4.00},
    }
    # 2) DADOS
    df = df_diff.copy()
    erros_concat = pd.concat([df[c] for c in df.columns], ignore_index=True).dropna()
    delta_h = erros_concat.values
    n = len(delta_h)

    if n == 0:
        escrever_no_arquivo("ERRO: Nenhum dado válido encontrado em df_diff.")
    else:
        # 3) CÁLCULOS

        rmse_z = np.sqrt(np.mean(delta_h**2))
        pec_pcd_z = 2 * rmse_z
        std_erro = np.std(delta_h, ddof=1)
        media_erro = np.mean(delta_h)

        # Intervalo de confiança para RMSE (via qui-quadrado)
        if n > 1:
            s2 = np.var(delta_h, ddof=1)
            chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n - 1)
            chi2_lower = stats.chi2.ppf(alpha/2, df=n - 1)
            rmse_lower = np.sqrt((n - 1) * s2 / chi2_upper) if chi2_upper > 0 else np.nan
            rmse_upper = np.sqrt((n - 1) * s2 / chi2_lower) if chi2_lower > 0 else np.nan
            pec_lower = 2 * rmse_lower
            pec_upper = 2 * rmse_upper
        else:
            rmse_lower = rmse_upper = pec_lower = pec_upper = np.nan

        # 4) CLASSIFICAÇÃO
        if escala not in limites_pec_pcd:
            escrever_no_arquivo(f"ERRO: Escala 1:{escala} não suportada.")
        else:
            limites = limites_pec_pcd[escala]
            classe_atendida = None
            conclusoes = {}

            for classe, rmse_limite in limites.items():
                if rmse_z <= rmse_limite:
                    classe_atendida = classe
                    break

            for classe, rmse_limite in limites.items():
                conclusoes[classe] = "ATENDE" if rmse_z <= rmse_limite else "NÃO ATENDE"

            # 5) ESCRITA NO ARQUIVO
            escrever_no_arquivo("===== AVALIAÇÃO DA PEC-PCD ALTIMÉTRICA (IN 1/2005) =====")
            escrever_no_arquivo(f"Escala do produto: 1:{escala}")
            escrever_no_arquivo(f"Média dos erros: {media_erro:+.4f} m")
            escrever_no_arquivo(f"Desvio padrão dos erros: {std_erro:.4f} m")
            escrever_no_arquivo(f"RMSE_z: {rmse_z:.4f} m")
            escrever_no_arquivo(f"PEC-PCD_z (2 × RMSE): {pec_pcd_z:.4f} m")
            
            if n > 1:
                escrever_no_arquivo(f"IC {(1-alpha)*100:.0f}% do RMSE_z: [{rmse_lower:.4f}, {rmse_upper:.4f}] m")
                escrever_no_arquivo(f"IC {(1-alpha)*100:.0f}% do PEC-PCD_z: [{pec_lower:.4f}, {pec_upper:.4f}] m")
            else:
                escrever_no_arquivo("IC não calculado (n ≤ 1).")
            
            escrever_no_arquivo("")
            escrever_no_arquivo("===== CLASSIFICAÇÃO POR CLASSE =====")
            
            for classe, status in conclusoes.items():
                rmse_max = limites[classe]
                pec_max = 2 * rmse_max
                escrever_no_arquivo(f"Classe {classe} → RMSE_z ≤ {rmse_max:.2f} m (PEC ≤ {pec_max:.2f} m) → {status}")
            
            classe_final = classe_atendida if classe_atendida else "Nenhuma"
            escrever_no_arquivo(f"\n=> Melhor classe atendida: {classe_final}")

    print("Terminou de rodar!")