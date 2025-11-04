from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from matplotlib.backends.backend_pdf import PdfPages

# GERAR RELATÒRIO A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import os


doc = SimpleDocTemplate(
    "Relatorio_Altimetrico.pdf",
    pagesize=A4,
    leftMargin=25,
    rightMargin=25,
    topMargin=40,
    bottomMargin=40
)


# ESTILOS DE TEXTO
estilos = getSampleStyleSheet()

estilos.add(ParagraphStyle(
    name='TituloCentral',
    parent=estilos['Title'],
    alignment=1,
    fontSize=18,
    spaceAfter=10
))

estilos.add(ParagraphStyle(
    name='LegendaImagem',
    parent=estilos['Normal'],
    alignment=1,
    fontSize=0,
    textColor=colors.black,
    spaceBefore=5,
    spaceAfter=10
))


# CONTEÚDO DO RELATÓRIO

conteudo = []
conteudo.append(Paragraph("Relatório Altimétrico", estilos["TituloCentral"]))
conteudo.append(Spacer(1, 20))


# LEITURA DAS IMAGENS 
pasta_imagens = "."  # Altere se quiser outra pasta
formatos_validos = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")

# Filtra apenas arquivos de imagem
imagens = [f for f in os.listdir(pasta_imagens) if f.lower().endswith(formatos_validos)]
def chave_ordenacao(nome):
    nome_lower = nome.lower()
    if 'perfil' in nome_lower:
        return 0
    elif 'boxplot' in nome_lower:
        return 1
    elif 'dist' in nome_lower:
        return 2
    else:
        return 3  # Outros vêm por último
imagens.sort(key=chave_ordenacao)

if not imagens:
    conteudo.append(Paragraph("⚠️ Nenhuma imagem encontrada.", estilos["Normal"]))
else:
    for i, img_nome in enumerate(imagens, start=1):
        caminho = os.path.join(pasta_imagens, img_nome)
        
        if os.path.exists(caminho):
            # Ajusta largura para quase toda a página (com margens)
            img = Image(caminho, width=500, height=200)# Ajuste altura conforme necessário
            img.hAlign = 'CENTER'  # Centraliza a imagem
            
            # Legenda limpa: apenas "Figura 1", "Figura 2", etc.
            legenda = Paragraph(f"Figura {i}", estilos["LegendaImagem"])
            
            conteudo.append(img)
            conteudo.append(legenda)
            conteudo.append(Spacer(1, 10))  # Espaço após cada bloco imagem+legenda
        else:
            conteudo.append(Paragraph(f" Imagem não carregada: {i}", estilos["Normal"]))
            conteudo.append(Spacer(1, 10))


# TESTE DE NORMALIDADE DO RELATÓRIO
# Adiciona título da seção
conteudo.append(PageBreak())  # quebra de página (remova se quiser no mesmo lugar)
conteudo.append(Paragraph("<b>Teste de Normalidade (Shapiro–Wilk)</b>", estilos["TituloCentral"]))
conteudo.append(Spacer(1, 10))

dados_tabela = [df_normalidade.columns.tolist()]

# Linhas formatadas com arredondamento
for _, linha in df_normalidade.iterrows():
    dados_tabela.append([
        linha["Coluna"],
        f"{linha['W']:.4f}" if pd.notna(linha["W"]) else "-",
        f"{linha['p']:.4f}" if pd.notna(linha["p"]) else "-",
        linha["Resultado"]
    ])

# Cria a tabela com largura proporcional
tabela = Table(dados_tabela, colWidths=[150, 100, 100, 150])

# Define o estilo visual da tabela
tabela.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
    ('TOPPADDING', (0, 0), (-1, 0), 6),
]))

# Adiciona a tabela ao conteúdo do relatório
conteudo.append(tabela)
conteudo.append(Spacer(1, 20))

# GERA O PDF FINAL

doc.build(conteudo)
print("Relatório gerado com sucesso!")