import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Barba de Elite - Dashboard de Vendas", layout="wide")

st.title("📈 Barba de Elite - Dashboard de Vendas e Previsões")
st.markdown("""
Este site permite que você envie seus arquivos de **estoque** e **vendas**, visualize os dados, veja a tendência de vendas diárias e ainda tenha acesso a **previsões dos próximos 30 dias** usando IA (modelo Prophet).
""")

# Upload de arquivos
st.sidebar.header("Upload de Arquivos")
estoque_file = st.sidebar.file_uploader("Envie o arquivo de estoque (.csv)", type="csv")
vendas_file = st.sidebar.file_uploader("Envie o arquivo de vendas (.csv)", type="csv")

if estoque_file and vendas_file:
    # Leitura dos dados
    df_estoque = pd.read_csv(estoque_file)
    df_vendas = pd.read_csv(vendas_file)

    # Preparo das datas
    df_vendas["Data e Hora"] = pd.to_datetime(df_vendas["Data e Hora"])
    df_vendas["Data"] = df_vendas["Data e Hora"].dt.date
    vendas_diarias = df_vendas.groupby("Data").size().reset_index(name="Vendas")

    # Mostrar dados
    with st.expander("🔍 Visualizar dados de estoque"):
        st.dataframe(df_estoque)

    with st.expander("🔍 Visualizar dados de vendas"):
        st.dataframe(df_vendas)

    # Gráfico de vendas diárias
    st.subheader("📊 Tendência de Vendas Diárias")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vendas_diarias["Data"],
        y=vendas_diarias["Vendas"],
        mode="lines+markers",
        name="Vendas Diárias",
        line=dict(color="blue")
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Previsão com Prophet
    st.subheader("🔮 Previsão de Vendas para os Próximos 30 Dias")
    df_prophet = vendas_diarias.rename(columns={"Data": "ds", "Vendas": "y"})
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão'))
    fig2.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Vendas Históricas'))

    st.plotly_chart(fig2, use_container_width=True)

    st.success("Análise concluída com sucesso!")

else:
    st.warning("Por favor, envie os dois arquivos .csv para visualizar a análise.")
