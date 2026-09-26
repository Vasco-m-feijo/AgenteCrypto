import os
import time
import json
import sqlite3
import warnings
from datetime import datetime

import requests
import pandas as pd
import numpy as np

# Suprime avisos secundários de bibliotecas no terminal
warnings.filterwarnings("ignore")

# Importação da API do Gemini
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# =============================================================
# CONFIGURAÇÕES GERAIS E TELEGRAM
# =============================================================
DB_NAME = "trading_agent.db"

# Cole seu Token e Chat ID aqui entre as aspas (ou configure nas variáveis de ambiente)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8232585341:AAH62WYlc3f2sDGiVmknoJFwnrqFoOLXx_g")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7335081779")

def send_telegram_alert(status_resultado, pnl_cash, pnl_pct, duration_min, new_balance, symbol, tf):
    """Envia uma notificação formatada para o seu celular via Telegram."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "SEU_TOKEN_AQUI":
        return  # Se não configurou o token, pula silenciosamente

    emoji = "✅" if pnl_cash > 0 else "❌"
    
    mensagem = (
        f"{emoji} <b>TRADE FINALIZADO ({symbol})</b>\n\n"
        f"• <b>Resultado:</b> {status_resultado}\n"
        f"• <b>Valor do Trade:</b> ${pnl_cash:+.2f} ({pnl_pct:+.2f}%)\n"
        f"• <b>Tempo do Trade:</b> {duration_min:.1f} minutos ({tf})\n"
        f"• <b>Saldo Final da Banca:</b> ${new_balance:.2f}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensagem,
        "parse_mode": "HTML"
    }

    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[AVISO] Falha ao enviar notificação no Telegram: {e}")

# =============================================================
# 1. BANCO DE DADOS E CARTEIRA VIRTUAL (SQLite)
# =============================================================
def init_db(initial_balance=1000.0):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS wallet (
        id INTEGER PRIMARY KEY,
        balance REAL,
        initial_balance REAL
    )
    """)
    cursor.execute("SELECT COUNT(*) FROM wallet")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO wallet (id, balance, initial_balance) VALUES (1, ?, ?)", (initial_balance, initial_balance))
    
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
        position_size REAL,
        exit_price REAL,
        exit_time TEXT,
        pnl_percent REAL,
        pnl_cash REAL,
        duration_minutes REAL,
        lesson TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_wallet():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT balance, initial_balance FROM wallet WHERE id = 1")
    balance, initial = cursor.fetchone()
    conn.close()
    return balance, initial

def update_wallet(amount_change):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE wallet SET balance = balance + ? WHERE id = 1", (amount_change,))
    conn.commit()
    conn.close()

# =============================================================
# 2. COLETA DE MERCADO E INDICADORES TÉCNICOS
# =============================================================
def get_klines_df(symbol="BTCUSDT", interval="5m", limit=60):
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
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - close.shift()).abs()
    low_close = (df["low"] - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    atr = ranges.max(axis=1).rolling(14).mean()

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

# =============================================================
# 3. RECUPERAÇÃO DE LIÇÕES DA MEMÓRIA
# =============================================================
def get_relevant_past_lessons(market_regime, limit=3):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT lesson FROM trades 
        WHERE lesson IS NOT NULL AND market_regime = ?
        ORDER BY id DESC LIMIT ?
    """, (market_regime, limit))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

# =============================================================
# 4. DECISOR DE IA (Gemini 3.8 Flash + Fallback)
# =============================================================
def run_ai_decision(market_data, sentiment, past_lessons):
    regime = f"{market_data['trend_ema']} | RSI {market_data['rsi_14']}"
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key and genai:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
Você é um gestor de risco de criptomoedas operando em modo de simulação com uma banca virtual.
Lições aprendidas anteriormente:
{past_lessons[-3:] if past_lessons else 'Nenhuma lição prévia.'}

Dados técnicos atuais:
- Preço do Bitcoin: ${market_data['current_price']}
- Tendência de Médias: {market_data['trend_ema']} (EMA20 vs EMA50)
- RSI (14): {market_data['rsi_14']}
- Volatilidade ATR: {market_data['atr_volatility']}
- Sentimento Geral de Mercado: {sentiment}

Defina a ação ('BUY', 'SELL' ou 'HOLD'). Responda ESTRITAMENTE em formato JSON:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "recommended_timeframe": "5m" | "15m" | "1h",
  "thesis": "justificativa sucinta",
  "target_profit_percent": 0.8,
  "stop_loss_percent": 0.5
}}
"""
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            decision = json.loads(response.text)
            decision["regime"] = regime
            return decision
        except Exception as e:
            print(f"[AVISO] Falha na API: {e}. Usando lógica de fallback.")

    # Fallback local
    if (market_data["trend_ema"] == "ALTA" and market_data["rsi_14"] < 55) or market_data["rsi_14"] < 40:
        return {
            "action": "BUY", "recommended_timeframe": "5m",
            "thesis": "Momento favorável de compra ou recuperação em suporte.",
            "target_profit_percent": 0.5, "stop_loss_percent": 0.4, "regime": regime
        }
    elif (market_data["trend_ema"] == "BAIXA" and market_data["rsi_14"] > 45) or market_data["rsi_14"] > 60:
        return {
            "action": "SELL", "recommended_timeframe": "5m",
            "thesis": "Pressão vendedora ou exaustão de curto prazo.",
            "target_profit_percent": 0.5, "stop_loss_percent": 0.4, "regime": regime
        }
    return {
        "action": "HOLD", "recommended_timeframe": "5m",
        "thesis": "Mercado em consolidação sem viés evidente.",
        "target_profit_percent": 0.0, "stop_loss_percent": 0.0, "regime": regime
    }

# =============================================================
# 5. REGISTRO E AVALIAÇÃO DE TRADES
# =============================================================
def record_trade(symbol, decision, current_price, position_size=100.0):
    # Evita abrir novos trades se já houver um em andamento
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'")
    if cur.fetchone()[0] >= 1:
        conn.close()
        return  # Já existe uma posição aberta sendo monitorada
    conn.close()
    if decision["action"] == "HOLD":
        print(f"[HOLD] {decision['thesis']}")
        return

    balance, _ = get_wallet()
    if balance < position_size:
        print(f"[SALDO INSUFICIENTE] Saldo de ${balance:.2f} menor que o tamanho da posição (${position_size:.2f}).")
        return

    update_wallet(-position_size)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trades (
            symbol, timeframe, action, entry_price, entry_time,
            thesis, market_regime, target_pnl, stop_loss, status, position_size
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
    """, (
        symbol,
        decision.get("recommended_timeframe", "5m"),
        decision["action"],
        current_price,
        datetime.now().isoformat(),
        decision["thesis"],
        decision.get("regime", ""),
        decision["target_profit_percent"],
        decision["stop_loss_percent"],
        position_size
    ))
    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()

    print(f"[NOVO TRADE #{trade_id}] {decision['action']} em {symbol} | Investido: ${position_size:.2f} a ${current_price:.2f} | Saldo em Caixa: ${balance - position_size:.2f}")

def evaluate_open_trades(current_price):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, timeframe, action, entry_price, entry_time, 
               thesis, market_regime, target_pnl, stop_loss, position_size 
        FROM trades WHERE status = 'OPEN'
    """)
    open_trades = cursor.fetchall()

    for row in open_trades:
        t_id, symbol, tf, action, entry, entry_time, thesis, regime, target_pnl, stop_pnl, pos_size = row
        
        diff_pct = ((current_price - entry) / entry) * 100 if action == "BUY" else ((entry - current_price) / entry) * 100

        target_reached = diff_pct >= target_pnl
        stop_reached = diff_pct <= -stop_pnl

        if target_reached or stop_reached:
            pnl_cash = pos_size * (diff_pct / 100.0)
            
            update_wallet(pos_size + pnl_cash)
            new_balance, _ = get_wallet()

            status = "CLOSED"
            status_resultado = "SUCESSO" if diff_pct > 0 else "PERDA"
            exit_time = datetime.now()
            entry_dt = datetime.fromisoformat(entry_time)
            duration_minutes = (exit_time - entry_dt).total_seconds() / 60.0

            if diff_pct > 0:
                lesson = f"Acerto [{regime} | {tf}]: Ganho de +${pnl_cash:.2f} (+{diff_pct:.2f}%) em {duration_minutes:.1f}min. Tese: {thesis}."
            else:
                lesson = f"Erro [{regime} | {tf}]: Perda de -${abs(pnl_cash):.2f} ({diff_pct:.2f}%) em {duration_minutes:.1f}min. Tese falhou: {thesis}."

            cursor.execute("""
                UPDATE trades 
                SET status = ?, exit_price = ?, exit_time = ?, pnl_percent = ?, pnl_cash = ?, duration_minutes = ?, lesson = ?
                WHERE id = ?
            """, (status, current_price, exit_time.isoformat(), round(diff_pct, 2), round(pnl_cash, 2), round(duration_minutes, 1), lesson, t_id))

            print(f"[TRADE ENCERRADO #{t_id}] Resultado: {status_resultado} de ${pnl_cash:+.2f} ({diff_pct:+.2f}%) | Novo Saldo: ${new_balance:.2f}")

            # DISPARA A NOTIFICAÇÃO NO CELULAR VIA TELEGRAM
            send_telegram_alert(
                status_resultado=status_resultado,
                pnl_cash=pnl_cash,
                pnl_pct=diff_pct,
                duration_min=duration_minutes,
                new_balance=new_balance,
                symbol=symbol,
                tf=tf
            )

    conn.commit()
    conn.close()

# =============================================================
# 6. RELATÓRIO NO TERMINAL
# =============================================================
def display_dashboard():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    balance, initial = get_wallet()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'")
    open_count = cursor.fetchone()[0]

    cursor.execute("SELECT pnl_percent, pnl_cash FROM trades WHERE status = 'CLOSED'")
    closed = cursor.fetchall()
    conn.close()

    print("\n" + "=" * 55)
    print(f" PAINEL DE CONTROLE DA BANCA SIMULADA")
    print(f" Saldo Disponível: ${balance:.2f} (Inicial: ${initial:.2f} | Variação Total: ${balance - initial:+.2f})")
    print(f" Operações Abertas: {open_count} | Operações Finalizadas: {len(closed)}")
    
    if closed:
        wins = [c for c in closed if c[0] > 0]
        win_rate = (len(wins) / len(closed)) * 100
        total_pnl_cash = sum(c for c in closed)
        print(f" Taxa de Acerto: {win_rate:.1f}% | Lucro Líquido Realizado: ${total_pnl_cash:+.2f}")
    print("=" * 55 + "\n")

# =============================================================
# 7. CICLO PRINCIPAL
# =============================================================
def run_pipeline(symbol="BTCUSDT"):
    init_db()
    
    # 1. Coleta dados (velas de 5 minutos)
    df_klines = get_klines_df(symbol, interval="5m", limit=60)
    tech = calculate_technical_indicators(df_klines)
    sentiment = get_fear_and_greed_index()
    
    # 2. Avalia ordens abertas (se bater alvo/stop, fecha e avisa no Telegram)
    evaluate_open_trades(tech["current_price"])

    # 3. Busca lições passadas relevantes na memória
    regime = f"{tech['trend_ema']} | RSI {tech['rsi_14']}"
    past_lessons = get_relevant_past_lessons(regime)

    # 4. Decisão da IA e registro
    decision = run_ai_decision(tech, sentiment, past_lessons)
    record_trade(symbol, decision, tech["current_price"], position_size=100.0)

    # 5. Exibe painel
    display_dashboard()

if __name__ == "__main__":
    INTERVALO_SEGUNDOS = 60  # Tempo entre cada verificação de mercado
    print("Iniciando agente em execução contínua. Pressione Ctrl+C para encerrar.")
    
    while True:
        try:
            run_pipeline("BTCUSDT")
            time.sleep(INTERVALO_SEGUNDOS)
        except requests.exceptions.RequestException as e:
            print(f"[AVISO] Oscilação momentânea de internet: {e}. Aguardando 15s...")
            time.sleep(15)
        except KeyboardInterrupt:
            print("\nAgente finalizado com segurança.")
            break