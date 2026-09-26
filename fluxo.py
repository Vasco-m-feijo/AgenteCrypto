import os
import sqlite3
import json
import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np

# Tenta carregar a biblioteca do Google GenAI se instalada
try:
    from google import genai
    from google.genai import types
except (ImportError, AttributeError):
    try:
        import google.genai as genai
        from google.genai import types
    except (ImportError, AttributeError):
        genai = None

DB_NAME = "trading_agent.db"

# -------------------------------------------------------------
# 1. BANCO DE DADOS LOCAL (SQLite)
# -------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Tabela de operações simuladas
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        timeframe TEXT,
        action TEXT,
        entry_price REAL,
        entry_time TEXT,
        thesis TEXT,
        market_regime TEXT,
        target_pnl REAL,
        stop_loss REAL,
        status TEXT,
        exit_price REAL,
        exit_time TEXT,
        pnl_percent REAL,
        duration_minutes REAL,
        lesson TEXT
    )
    """)
    conn.commit()
    conn.close()

# -------------------------------------------------------------
# 2. COLETA DE DADOS ENRIQUECIDOS (Binance + Sentimento)
# -------------------------------------------------------------
def get_klines_df(symbol="BTCUSDT", interval="1h", limit=50):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url, timeout=10).json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def calculate_technical_indicators(df):
    close = df["close"]
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # Médias Móveis Exponenciais
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    # Volatilidade (ATR de 14 períodos)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - close.shift()).abs()
    low_close = (df["low"] - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(14).mean()

    return {
        "current_price": float(close.iloc[-1]),
        "rsi_14": round(float(rsi.iloc[-1]), 2),
        "ema_20": round(float(ema_20.iloc[-1]), 2),
        "ema_50": round(float(ema_50.iloc[-1]), 2),
        "trend_ema": "ALTA" if ema_20.iloc[-1] > ema_50.iloc[-1] else "BAIXA",
        "atr_volatility": round(float(atr.iloc[-1]), 2)
    }

def get_fear_and_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
        item = res["data"][0]
        return f"{item['value']} ({item['value_classification']})"
    except Exception:
        return "Indisponível"

# -------------------------------------------------------------
# 3. RECUPERAÇÃO DE LIÇÕES RELEVANTES (Filtro por Contexto)
# -------------------------------------------------------------
def get_relevant_past_lessons(market_regime, limit=3):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Prioriza lições de trades com regime de mercado semelhante
    cursor.execute("""
        SELECT lesson FROM trades 
        WHERE lesson IS NOT NULL AND market_regime = ?
        ORDER BY id DESC LIMIT ?
    """, (market_regime, limit))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

# -------------------------------------------------------------
# 4. DECISOR (IA com Contexto Otimizado)
# -------------------------------------------------------------
def run_ai_decision(market_data, sentiment, past_lessons):
    regime = f"{market_data['trend_ema']} | RSI {market_data['rsi_14']}"
    lessons_formatted = "\n".join([f"- {l}" for l in past_lessons]) if past_lessons else "Nenhuma lição anterior para este regime."

    prompt = f"""
Você é um agente de análise de criptomoedas em ambiente de simulação.
Lições aprendidas em cenários semelhantes:
{lessons_formatted}

Indicadores atuais:
- Preço: ${market_data['current_price']}
- Tendência (EMA20/50): {market_data['trend_ema']}
- RSI (14): {market_data['rsi_14']}
- Volatilidade (ATR): {market_data['atr_volatility']}
- Sentimento Geral de Mercado: {sentiment}

Determine a melhor ação e o horizonte operacional recomendado ('15m', '1h' ou '4h').
Responda ESTRITAMENTE em formato JSON:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "recommended_timeframe": "15m" | "1h" | "4h",
  "thesis": "justificativa sucinta",
  "target_profit_percent": 1.5,
  "stop_loss_percent": 0.8
}}
"""

    api_key = os.getenv("GEMINI_API_KEY")

    # Se a chave existir e a biblioteca estiver disponível, chama o LLM real
    if api_key and genai:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            decision = json.loads(response.text)
            decision["regime"] = regime
            return decision
        except Exception as e:
            print(f"[AVISO] Falha na chamada da API: {e}. Usando lógica de fallback.")

    # FALLBACK / SIMULAÇÃO (enquanto a chave não estiver configurada)
    if market_data["rsi_14"] < 32 and market_data["trend_ema"] == "ALTA":
        return {
            "action": "BUY",
            "recommended_timeframe": "1h",
            "thesis": "Correção com RSI sobrevendido a favor da tendência.",
            "target_profit_percent": 2.0,
            "stop_loss_percent": 1.0,
            "regime": regime
        }
    elif market_data["rsi_14"] > 70:
        return {
            "action": "SELL",
            "recommended_timeframe": "15m",
            "thesis": "RSI sobrecomprado indicando exaustão.",
            "target_profit_percent": 1.5,
            "stop_loss_percent": 0.8,
            "regime": regime
        }
    return {
        "action": "HOLD",
        "recommended_timeframe": "1h",
        "thesis": "Condições neutras, aguardando melhor relação risco/retorno.",
        "target_profit_percent": 0.0,
        "stop_loss_percent": 0.0,
        "regime": regime
    }

# -------------------------------------------------------------
# 5. REGISTRO E AVALIAÇÃO DE RESULTADOS
# -------------------------------------------------------------
def record_trade(symbol, decision, current_price):
    if decision["action"] == "HOLD":
        print(f"[HOLD] {decision['thesis']}")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trades (symbol, timeframe, action, entry_price, entry_time, thesis, market_regime, target_pnl, stop_loss, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (
        symbol,
        decision["recommended_timeframe"],
        decision["action"],
        current_price,
        datetime.now().isoformat(),
        decision["thesis"],
        decision["regime"],
        decision["target_profit_percent"],
        decision["stop_loss_percent"]
    ))
    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()
    print(f"[NOVO TRADE SIMULADO #{trade_id}] {decision['action']} em {symbol} ({decision['recommended_timeframe']}) a ${current_price:.2f}")

def evaluate_open_trades(current_price):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol, timeframe, action, entry_price, entry_time, thesis, market_regime, target_pnl, stop_loss FROM trades WHERE status = 'OPEN'")
    open_trades = cursor.fetchall()

    for row in open_trades:
        t_id, symbol, tf, action, entry, entry_time, thesis, regime, target_pnl, stop_pnl = row
        diff = ((current_price - entry) / entry) * 100 if action == "BUY" else ((entry - current_price) / entry) * 100

        target_reached = diff >= target_pnl
        stop_reached = diff <= -stop_pnl

        if target_reached or stop_reached:
            status = "CLOSED"
            exit_time = datetime.now()
            entry_dt = datetime.fromisoformat(entry_time)
            duration_minutes = (exit_time - entry_dt).total_seconds() / 60.0
            
            # Geração da lição reflexiva
            if diff > 0:
                lesson = f"Acerto [{regime} | {tf}]: Entrada validada com {diff:.2f}% em {duration_minutes:.1f}min. Tese: {thesis}."
            else:
                lesson = f"Erro [{regime} | {tf}]: Entrada falhou com {diff:.2f}% em {duration_minutes:.1f}min. Evitar operações sob essa configuração sem sinal de volume."

            cursor.execute("""
                UPDATE trades 
                SET status = ?, exit_price = ?, exit_time = ?, pnl_percent = ?, duration_minutes = ?, lesson = ?
                WHERE id = ?
            """, (status, current_price, exit_time.isoformat(), round(diff, 2), round(duration_minutes, 1), lesson, t_id))
            print(f"[TRADE ENCERRADO #{t_id}] PnL: {diff:.2f}% em {duration_minutes:.1f} min | Lição: {lesson}")

    conn.commit()
    conn.close()

# -------------------------------------------------------------
# 6. PAINEL DE MÉTRICAS E DESEMPENHO
# -------------------------------------------------------------
def display_performance_dashboard():
    """
    Consulta o banco SQLite trading_agent.db e imprime no terminal
    um painel visual limpo com métricas consolidadas de desempenho:
    - Posições em aberto (OPEN)
    - Total de operações fechadas (CLOSED)
    - Operações vencedoras (PnL > 0) e perdedoras (PnL <= 0)
    - Taxa de acerto (Win Rate %) formatada com 1 casa decimal
    - Lucro/Prejuízo acumulado (soma de PnL %) com sinal (+/-)
    - Duração média das operações em minutos
    - As 3 lições reflexivas mais recentes registradas
    """
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Contagem de operações abertas
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'")
    open_trades_count = cursor.fetchone()[0]

    # Métricas agregadas de operações fechadas
    cursor.execute("""
        SELECT 
            COUNT(*),
            SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN pnl_percent <= 0 THEN 1 ELSE 0 END),
            SUM(pnl_percent),
            AVG(duration_minutes)
        FROM trades 
        WHERE status = 'CLOSED'
    """)
    row = cursor.fetchone()
    total_closed = row[0] or 0
    wins = row[1] or 0
    losses = row[2] or 0
    total_pnl = row[3] or 0.0
    avg_duration = row[4] or 0.0

    # Taxa de acerto (Win Rate %)
    win_rate = (wins / total_closed * 100.0) if total_closed > 0 else 0.0

    # Busca as 3 lições mais recentes gravadas no campo lesson
    cursor.execute("""
        SELECT lesson 
        FROM trades 
        WHERE status = 'CLOSED' AND lesson IS NOT NULL AND TRIM(lesson) != ''
        ORDER BY id DESC 
        LIMIT 3
    """)
    recent_lessons = [r[0] for r in cursor.fetchall()]
    conn.close()

    # Exibição formatada em quadro textual limpo
    print("\n" + "=" * 70)
    print("                     PAINEL DE DESEMPENHO")
    print("=" * 70)
    print(f"  * Posições em Aberto (OPEN)       : {open_trades_count}")
    print(f"  * Total de Operações Fechadas     : {total_closed}")
    print(f"  * Operações Vencedoras (PnL > 0)  : {wins}")
    print(f"  * Operações Perdedoras (PnL <= 0) : {losses}")
    print(f"  * Taxa de Acerto (Win Rate)       : {win_rate:.1f}%")
    print(f"  * Lucro/Prejuízo Acumulado (PnL)  : {total_pnl:+.2f}%")
    print(f"  * Duração Média das Operações     : {avg_duration:.1f} min")
    print("-" * 70)
    print("  LIÇÕES RECENTES APRENDIDAS (Últimas 3):")
    if recent_lessons:
        for idx, lesson in enumerate(recent_lessons, start=1):
            print(f"    [{idx}] {lesson}")
    else:
        print("    (Nenhuma lição registrada até o momento)")
    print("=" * 70 + "\n")

# -------------------------------------------------------------
# 7. CICLO PRINCIPAL DO PIPELINE
# -------------------------------------------------------------
def run_pipeline(symbol="BTCUSDT"):
    init_db()
    
    # 1. Coleta dados
    df_klines = get_klines_df(symbol, interval="1h", limit=60)
    tech = calculate_technical_indicators(df_klines)
    sentiment = get_fear_and_greed_index()
    
    # 2. Avalia posições abertas com o preço atual
    evaluate_open_trades(tech["current_price"])

    # 3. Busca lições passadas relevantes para o regime atual
    regime = f"{tech['trend_ema']} | RSI {tech['rsi_14']}"
    past_lessons = get_relevant_past_lessons(regime)

    # 4. Decisão e Registro
    decision = run_ai_decision(tech, sentiment, past_lessons)
    record_trade(symbol, decision, tech["current_price"])

# -------------------------------------------------------------
# 8. CICLO CONTÍNUO COM TRATAMENTO DE PARADA SUAVE
# -------------------------------------------------------------
def run_continuous_agent(symbol="BTCUSDT", interval_minutes=5):
    """
    Executa o agente em loop contínuo (while True) agendado com time.sleep.
    Em cada iteração:
      1. Executa run_pipeline(symbol)
      2. Exibe display_performance_dashboard()
      3. Aguarda o intervalo definido em minutos
    
    Tratamento de exceções:
      - requests.RequestException: captura falhas transitórias de conexão de rede
      - KeyboardInterrupt (Ctrl + C): saída suave sem traceback
    """
    print("=" * 70)
    print("          INICIANDO AGENTE DE TRADING EM MODO CONTÍNUO")
    print(f"  * Par de negociação : {symbol}")
    print(f"  * Intervalo do ciclo: {interval_minutes} minuto(s)")
    print("  * Pressione Ctrl + C para encerrar a execução a qualquer momento.")
    print("=" * 70)

    try:
        while True:
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n>>> [CICLO INICIADO EM {timestamp}] Analisando {symbol}...")

            try:
                run_pipeline(symbol)
                display_performance_dashboard()
            except requests.RequestException as req_err:
                print(f"\n[ALERTA DE CONEXÃO] Falha de comunicação de rede: {req_err}")
                print("O agente permanecerá ativo e tentará novamente no próximo ciclo.")
            except Exception as err:
                print(f"\n[ERRO NO CICLO] Ocorreu um erro durante a execução: {err}")

            print(f"[AGUARDANDO] Próxima verificação em {interval_minutes} minuto(s)...")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("  [FINALIZAÇÃO] Execução interrompida pelo usuário (Ctrl + C).")
        print("  Encerrando o agente de trading com segurança. Até logo!")
        print("=" * 70)

# -------------------------------------------------------------
# PONTO DE ENTRADA
# -------------------------------------------------------------
if __name__ == "__main__":
    run_continuous_agent("BTCUSDT", interval_minutes=5)