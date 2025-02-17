import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioBacktester:
    def __init__(self, exchange_id, symbols, timeframe="1d", short_window=50, long_window=200, 
                 initial_balance=10000, trading_fee=0.001, start_date=None, end_date=None):
        self.exchange = getattr(ccxt, exchange_id)()
        self.symbols = symbols
        self.timeframe = timeframe
        self.short_window = short_window
        self.long_window = long_window
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.data = {}
        self.balance = initial_balance
        self.positions = {symbol: 0 for symbol in symbols}

    def fetch_data(self, limit=1000):
        """
        거래소 데이터를 가져오고 기간 필터링을 적용한다.
        """
        for symbol in self.symbols:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                if self.start_date and self.end_date:
                    df = df[(df['timestamp'] >= self.start_date) & (df['timestamp'] <= self.end_date)]

                df.set_index("timestamp", inplace=True)
                self.data[symbol] = df
                print(f"✅ {symbol}: {len(df)} records fetched")

            except Exception as e:
                print(f"🚨 API Error fetching {symbol}: {e}")

    def apply_strategy(self):
        """
        전략을 적용한다. (초기 NaN 데이터 제거)
        """
        for symbol in self.symbols:
            df = self.data[symbol]
            df["SMA_short"] = df["close"].rolling(window=self.short_window).mean()
            df["SMA_long"] = df["close"].rolling(window=self.long_window).mean()

            # 🛠️ 초기 NaN 데이터 제거
            df.dropna(inplace=True)

            df["Signal"] = 0
            df.loc[df["SMA_short"] > df["SMA_long"], "Signal"] = 1  # 매수 신호
            df.loc[df["SMA_short"] < df["SMA_long"], "Signal"] = -1 # 매도 신호


    def run_backtest(self):
        """
        백테스트 실행을 한다.
        """
        portfolio_value = self.initial_balance
        portfolio_history = []

        for date in self.data[self.symbols[0]].index:
            daily_value = 0
            for symbol in self.symbols:
                if date not in self.data[symbol].index:
                    continue
                
                signal = self.data[symbol].loc[date, "Signal"]
                price = self.data[symbol].loc[date, "close"]

                if signal == 1 and self.positions[symbol] == 0:  # 매수
                    self.positions[symbol] = (portfolio_value / len(self.symbols)) / price
                    portfolio_value -= (portfolio_value / len(self.symbols)) * self.trading_fee

                elif signal == -1 and self.positions[symbol] > 0:  # 매도
                    portfolio_value += self.positions[symbol] * price
                    portfolio_value -= (self.positions[symbol] * price) * self.trading_fee
                    self.positions[symbol] = 0

                daily_value += self.positions[symbol] * price

            portfolio_history.append([date, portfolio_value + daily_value])

        portfolio_df = pd.DataFrame(portfolio_history, columns=["timestamp", "Portfolio_Value"])
        portfolio_df.set_index("timestamp", inplace=True)

        return portfolio_df

    def calculate_mdd(self, portfolio_df):
        """
        최대 낙폭(MDD)을 계산한다.
        """
        peak = portfolio_df["Portfolio_Value"].cummax()
        drawdown = (portfolio_df["Portfolio_Value"] - peak) / peak
        return drawdown.min() * 100  

    def calculate_total_return(self, portfolio_df):
        """
        총 수익률을 계산한다.
        """
        return (portfolio_df["Portfolio_Value"].iloc[-1] / self.initial_balance - 1) * 100
    
    def calculate_monthly_returns(self, portfolio_df):
        """
        월별 수익률을 계산한다. (첫 달 수익률 포함)
        """
        monthly_values = portfolio_df["Portfolio_Value"].resample("M").last()
        
        first_month = monthly_values.index[0]
        first_month_return = (monthly_values.iloc[0] / self.initial_balance - 1) * 100

        monthly_returns = ((monthly_values / monthly_values.shift(1)) - 1) * 100
        monthly_returns.loc[first_month] = first_month_return
        monthly_returns = monthly_returns.sort_index()

        return monthly_returns.map(lambda x: f"{x:.2f}%")

    
    def calculate_cagr(self, portfolio_df):
        """
        연평균 수익률을 (CAGR) 계산한다.
        """
        start_value = self.initial_balance
        end_value = portfolio_df["Portfolio_Value"].iloc[-1]
        num_years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.0

        if num_years <= 0:
            return "N/A"

        cagr = ((end_value / start_value) ** (1 / num_years)) - 1
        return f"{cagr * 100:.2f}%"


    def plot_results(self, portfolio_df):
        """
        결과 시각화를 한다.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df["Portfolio_Value"], label="Portfolio Value", color="blue")
        plt.title("Portfolio Backtest - Moving Average Crossover")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid()
        plt.show()

    def compare_with_benchmark(self, portfolio_df):
        """
        비트코인 단독 투자와 비교한다.
        """
        btc_df = self.data["BTC/USDT"].copy()
        btc_df["BTC_Benchmark"] = (btc_df["close"] / btc_df["close"].iloc[0]) * self.initial_balance
        portfolio_df = portfolio_df.join(btc_df["BTC_Benchmark"], how="left")

        plt.figure(figsize=(12,6))
        plt.plot(portfolio_df.index, portfolio_df["Portfolio_Value"], label="Portfolio", color="blue")
        plt.plot(portfolio_df.index, portfolio_df["BTC_Benchmark"], label="BTC Only", color="orange", linestyle="dashed")
        plt.title("Portfolio vs. BTC Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    print("portfolio backtest start")

    symbols = ["BTC/USDT", "ETH/USDT"]
    start_date = "2023-01-01"
    end_date = "2024-12-31"

    backtester = PortfolioBacktester(
        exchange_id="binance",
        symbols=symbols,
        timeframe="1d",
        start_date=start_date,
        end_date=end_date
    )

    backtester.fetch_data()

    backtester.apply_strategy()
    portfolio_df = backtester.run_backtest()

    mdd = backtester.calculate_mdd(portfolio_df)
    cagr = backtester.calculate_cagr(portfolio_df)
    total_return = backtester.calculate_total_return(portfolio_df)
    monthly_returns = backtester.calculate_monthly_returns(portfolio_df)

    print("\n📊 Monthly Returns:")
    print(monthly_returns)

    print(f"📉 Maximum Drawdown (MDD): {mdd:.2f}%")
    print(f"📈 Total Return (ROI)): {total_return:.2f}%")
    print(f"📊 Compound Annual Growth Rate (CAGR): {cagr}")

    # 시각화
    # backtester.compare_with_benchmark(portfolio_df)
    print("portfolio backtest end")