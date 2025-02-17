import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioBacktester:
    def __init__(self, exchange_id, symbols, weights, timeframe="1d", initial_balance=10000, 
                 start_date=None, end_date=None, rebalance_period="1M", trading_fee=0.001, slippage=0.0005):
        self.exchange = getattr(ccxt, exchange_id)()
        self.symbols = symbols
        self.weights = weights
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.rebalance_period = rebalance_period.replace("M", "MS")
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.data = {}

    def fetch_data(self, limit=2000):
        """
        거래소 데이터를 가져온다.
        """
        for symbol in self.symbols:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
                if not ohlcv:
                    raise ValueError(f"No data found for {symbol}")

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                if self.start_date and self.end_date:
                    df = df[(df['timestamp'] >= self.start_date) & (df['timestamp'] <= self.end_date)]

                df.set_index("timestamp", inplace=True)
                self.data[symbol] = df
                print(f"✅ {symbol}: {len(df)} records fetched")

            except Exception as e:
                print(f"API Error fetching {symbol}: {e}")

    def calculate_portfolio_returns(self):
        """
        수익율 계산을 한다.
        """
        portfolio_df = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date, freq="D"))

        for symbol in self.symbols:
            df = self.data[symbol].copy()
            df[f"{symbol}_return"] = df["close"].pct_change()
            portfolio_df = portfolio_df.join(df[f"{symbol}_return"], how="left")

        portfolio_df.fillna(0, inplace=True)
        portfolio_df["Portfolio_Return"] = sum(portfolio_df[f"{symbol}_return"] * self.weights[symbol] for symbol in self.symbols)
        return portfolio_df
        
    def calculate_monthly_returns(self, portfolio_df):
        """
        월별 수익률을 계산하여 반환
        """
        monthly_values = portfolio_df["Portfolio_Value"].resample("M").last()
        monthly_returns = monthly_values.pct_change().dropna()

        return monthly_returns.map(lambda x: f"{x*100:.2f}%")


    def apply_trading_fees(self, portfolio_df):
        """
        수수료 적용을 한다.
        """
        transaction_cost = (self.trading_fee + self.slippage) * portfolio_df["Portfolio_Value"]
        portfolio_df["Portfolio_Value"] -= transaction_cost
        return portfolio_df

    def run_backtest(self):
        """
        백테스트를 한다.
        """
        portfolio_df = self.calculate_portfolio_returns()
        portfolio_df["Portfolio_Value"] = float(self.initial_balance)

        rebalance_dates = portfolio_df.resample(self.rebalance_period).first().index

        for i in range(1, len(portfolio_df)):
            date = portfolio_df.index[i]
            daily_return = portfolio_df.iloc[i]["Portfolio_Return"]

            if date in rebalance_dates:
                current_prices = {symbol: self.data[symbol]["close"].reindex(portfolio_df.index).ffill().loc[date] for symbol in self.symbols}
                if any(np.isnan(price) for price in current_prices.values()):
                    continue

                total_value = sum(current_prices[symbol] * self.weights[symbol] for symbol in self.symbols)
                for symbol in self.symbols:
                    self.weights[symbol] = (current_prices[symbol] * self.weights[symbol]) / total_value

            portfolio_df.loc[date, "Portfolio_Value"] = float(portfolio_df.iloc[i - 1]["Portfolio_Value"]) * (1 + daily_return)

        portfolio_df = self.apply_trading_fees(portfolio_df)
        return portfolio_df

    def calculate_mdd(self, portfolio_df):
        """
        최대 낙폭을 계산한다.
        """
        peak = portfolio_df["Portfolio_Value"].cummax()
        drawdown = (portfolio_df["Portfolio_Value"] - peak) / peak
        return drawdown.min() * 100  

    def calculate_total_return(self, portfolio_df):
        """
        총 수익율을 계산한다.
        """
        return (portfolio_df["Portfolio_Value"].iloc[-1] / self.initial_balance - 1) * 100

    def compare_with_benchmark(self, portfolio_df):
        """
        비트코인 벤치마크를 해본다.
        """
        btc_df = self.data["BTC/USDT"].copy()
        btc_df["BTC_Benchmark"] = (1 + btc_df["close"].pct_change()).cumprod() * self.initial_balance
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

    def plot_portfolio_value(self, portfolio_df):
        """
        시각화를 한다.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df["Portfolio_Value"], label="Portfolio Value", color="blue")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    print("portfolio backtest start")
    
    symbols = ["BTC/USDT", "ETH/USDT", "DOT/USDT", "USDC/USDT"]
    weights = {"BTC/USDT": 0.4, "ETH/USDT": 0.2, "DOT/USDT": 0.2, "USDC/USDT": 0.2}

    start_date = "2022-01-01"
    end_date = "2025-01-01"
    rebalance_period = "1M"

    backtester = PortfolioBacktester(exchange_id="binance", symbols=symbols, weights=weights, timeframe="1M", 
                                     start_date=start_date, end_date=end_date, rebalance_period=rebalance_period)

    backtester.fetch_data()
    portfolio_df = backtester.run_backtest()

    mdd = backtester.calculate_mdd(portfolio_df)
    total_return = backtester.calculate_total_return(portfolio_df)
    monthly_returns = backtester.calculate_monthly_returns(portfolio_df)

    print("\n📊 Monthly Returns:")
    print(monthly_returns)

    print(f"📉 Maximum Drawdown (MDD): {mdd:.2f}%")
    print(f"📈 Total Return: {total_return:.2f}%")

    # backtester.compare_with_benchmark(portfolio_df)    
    print("portfolio backtest end")
