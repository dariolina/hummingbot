"""
Standalone backtest script for RSI Trend Follower strategy.

This script runs a true backtest on historical data without requiring the strategy to run live.
It simulates the strategy logic on past candles and calculates performance.

Usage:
    conda activate hummingbot
    python scripts/utility/backtest_rsi_trend_follower_standalone.py
"""

import sys
import os

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change to project root directory
if os.path.exists(os.path.join(project_root, 'hummingbot')):
    os.chdir(project_root)

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta

from hummingbot import data_path
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HistoricalCandlesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestResult:
    """Represents a single trade result."""
    def __init__(self, entry_time: float, entry_price: float, side: str, amount: float):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.side = side  # "BUY" or "SELL"
        self.amount = amount
        self.exit_time: Optional[float] = None
        self.exit_price: Optional[float] = None
        self.close_type: Optional[str] = None
        self.pnl_pct: float = 0.0
        self.pnl_quote: float = 0.0
        self.fees_paid: float = 0.0
        self.is_open = True
        self.effective_time_limit: Optional[float] = None  # Can be extended if profitable
        self.time_limit_extended: bool = False  # Track if we've already extended once


class RSITrendFollowerBacktest:
    """Standalone backtest engine for RSI Trend Follower strategy."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trades: List[BacktestResult] = []
        self.current_position: Optional[BacktestResult] = None
        self.candles_df: Optional[pd.DataFrame] = None
        self.last_position_close_time: Optional[float] = None
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        if interval.endswith('m'):
            return int(interval[:-1]) * 60
        elif interval.endswith('h'):
            return int(interval[:-1]) * 3600
        elif interval.endswith('d'):
            return int(interval[:-1]) * 86400
        elif interval.endswith('s'):
            return int(interval[:-1])
        else:
            raise ValueError(f"Unknown interval format: {interval}")
        
    async def load_historical_candles(self, start_time: int, end_time: int):
        """Load historical candles for the backtest period."""
        logger.info(f"Loading historical candles from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
        
        # Calculate max_records based on interval and time period
        # Add buffer for safety
        time_span = end_time - start_time
        interval_seconds = self._interval_to_seconds(self.config['candles_interval'])
        estimated_candles = int(time_span / interval_seconds) + 1000  # Add 1000 buffer
        
        candle = CandlesFactory.get_candle(CandlesConfig(
            connector=self.config['candles_exchange'],
            trading_pair=self.config['trading_pair'],
            interval=self.config['candles_interval'],
            max_records=max(estimated_candles, 10000)  # At least 10000, or calculated amount
        ))
        
        # Fetch historical candles
        self.candles_df = await candle.get_historical_candles(HistoricalCandlesConfig(
            connector_name=self.config['candles_exchange'],
            trading_pair=self.config['trading_pair'],
            interval=self.config['candles_interval'],
            start_time=start_time,
            end_time=end_time
        ))
        
        logger.info(f"Loaded {len(self.candles_df)} candles")
        return self.candles_df
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator."""
        df = df.copy()
        df.ta.rsi(length=self.config['rsi_length'], append=True)
        return df
    
    def get_signal(self, df: pd.DataFrame, idx: int) -> int:
        """Get trading signal at index idx."""
        if idx < self.config['rsi_lookback_candles']:
            return 0
        
        rsi_column = f"RSI_{self.config['rsi_length']}"
        if rsi_column not in df.columns:
            return 0
        
        current_rsi = df[rsi_column].iloc[idx]
        lookback_rsi = df[rsi_column].iloc[idx - self.config['rsi_lookback_candles']]
        previous_rsi = df[rsi_column].iloc[idx - 1] if idx > 0 else current_rsi
        
        threshold = self.config['rsi_direction_threshold']
        
        signal = 0
        is_counter_trend = self.config.get('counter_trend_mode', False)
        
        # Extreme RSI signals - high probability reversals regardless of trend
        # These bypass the normal threshold requirement
        extreme_rsi_overbought = self.config.get('extreme_rsi_overbought', 85.0)
        extreme_rsi_oversold = self.config.get('extreme_rsi_oversold', 15.0)
        enable_extreme_rsi = self.config.get('enable_extreme_rsi_signals', True)
        
        if enable_extreme_rsi:
            # Extreme overbought → SHORT (expecting reversal down)
            if current_rsi >= extreme_rsi_overbought:
                return -1
            # Extreme oversold → LONG (expecting reversal up)
            if current_rsi <= extreme_rsi_oversold:
                return 1

        if is_counter_trend:
            # Counter-trend: Bet against the momentum (mean reversion)
            # SHORT when RSI is rising a lot (expecting reversal down)
            # LONG when RSI is dropping a lot (expecting reversal up)
            if current_rsi >= lookback_rsi + threshold:
                # RSI is rising strongly - bet SHORT (expecting reversal)
                signal = -1
            elif current_rsi <= lookback_rsi - threshold:
                # RSI is dropping strongly - bet LONG (expecting reversal)
                signal = 1
        else:
            # Trend-following: Enter LONG when RSI is rising, SHORT when RSI is dropping
            # SHORT signal
            if current_rsi <= lookback_rsi - threshold and current_rsi < previous_rsi:
                signal = -1
            # LONG signal
            elif current_rsi >= lookback_rsi + threshold and current_rsi > previous_rsi:
                signal = 1
        
        return signal
    
    def check_market_condition(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if market condition filter passes."""
        if not self.config.get('enable_market_condition_filter', False):
            return True
        
        try:
            # Calculate ADX
            adx = ta.adx(
                df["high"].iloc[:idx+1],
                df["low"].iloc[:idx+1],
                df["close"].iloc[:idx+1],
                length=self.config.get('adx_length', 14)
            )
            adx_column = f"ADX_{self.config.get('adx_length', 14)}"
            
            if adx_column in adx.columns and len(adx) > 0:
                current_adx = adx[adx_column].iloc[-1]
                # Both trend-following and counter-trend need trending markets (high ADX)
                # Counter-trend just trades the opposite direction, but still needs a trend to exist
                if current_adx < self.config.get('min_adx_trend_strength', 25.0):
                    return False
            
            # Check RSI variance
            rsi_column = f"RSI_{self.config['rsi_length']}"
            if rsi_column in df.columns:
                lookback = self.config.get('rsi_variance_lookback', 10)
                if idx >= lookback:
                    rsi_values = df[rsi_column].iloc[idx-lookback+1:idx+1]
                    rsi_variance = rsi_values.var()
                    if rsi_variance > self.config.get('max_rsi_variance', 400.0):
                        return False
            
            return True
        except:
            return True
    
    def is_extreme_rsi_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current candle has extreme RSI (bypasses other filters)."""
        if not self.config.get('enable_extreme_rsi_signals', True):
            return False
        
        rsi_column = f"RSI_{self.config['rsi_length']}"
        if rsi_column not in df.columns:
            return False
        
        current_rsi = df[rsi_column].iloc[idx]
        extreme_overbought = self.config.get('extreme_rsi_overbought', 85.0)
        extreme_oversold = self.config.get('extreme_rsi_oversold', 15.0)
        
        return current_rsi >= extreme_overbought or current_rsi <= extreme_oversold
    
    def check_filters(self, df: pd.DataFrame, idx: int, signal: int) -> bool:
        """Check if signal passes all filters."""
        rsi_column = f"RSI_{self.config['rsi_length']}"
        if rsi_column not in df.columns:
            return False
        
        current_rsi = df[rsi_column].iloc[idx]
        
        # Check if this is an extreme RSI signal (bypasses RSI threshold and market condition filters, but NOT volatility)
        is_extreme = self.is_extreme_rsi_signal(df, idx)
        
        if not is_extreme:
            # Non-extreme signals need to pass RSI threshold and market condition filters
            
            # Check if we're in counter-trend mode (signal flipped)
            # If signal is flipped, we need to flip the filters too
            # Counter-trend: LONG when RSI high (overbought), SHORT when RSI low (oversold)
            is_counter_trend = self.config.get('counter_trend_mode', False)
            
            # RSI threshold filters
            if is_counter_trend:
                # Counter-trend: SHORT when RSI is rising (expecting reversal), LONG when RSI is dropping (expecting reversal)
                # We WANT to SHORT when RSI is high (overbought) and LONG when RSI is low (oversold)
                # So we filter SHORT when RSI is too low, LONG when RSI is too high
                if signal == -1 and current_rsi < self.config.get('rsi_oversold', 30.0):
                    return False  # Don't SHORT when RSI is too low (oversold) in counter-trend - we want to short when RSI is high
                if signal == 1 and current_rsi > self.config.get('rsi_overbought', 70.0):
                    return False  # Don't LONG when RSI is too high (overbought) in counter-trend - we want to long when RSI is low
            else:
                # Trend-following: LONG when RSI rising, SHORT when RSI dropping
                # Filter LONG when overbought, SHORT when oversold
                if signal == 1 and current_rsi > self.config.get('rsi_overbought', 70.0):
                    return False
                if signal == -1 and current_rsi < self.config.get('rsi_oversold', 30.0):
                    return False
            
            # Market condition filter (only for non-extreme signals)
            if not self.check_market_condition(df, idx):
                return False
        
        # Volatility filter (applies to ALL signals, including extreme RSI)
        if self.config.get('enable_volatility_filter', False):
            try:
                natr = ta.natr(
                    df["high"].iloc[:idx+1],
                    df["low"].iloc[:idx+1],
                    df["close"].iloc[:idx+1],
                    length=self.config.get('atr_length', 14)
                )
                if "NATR" in natr.columns and len(natr) > 0:
                    natr_value = natr["NATR"].iloc[-1]
                    if natr_value < self.config.get('min_volatility_pct', 0.15):
                        return False
            except:
                pass
        
        return True
    
    def check_exit_conditions(self, df: pd.DataFrame, idx: int, position: BacktestResult) -> Optional[str]:
        """Check if position should be closed."""
        if not position.is_open:
            return None
        
        entry_price = position.entry_price
        candle_high = df['high'].iloc[idx]
        candle_low = df['low'].iloc[idx]
        candle_close = df['close'].iloc[idx]
        
        # Use static TP/SL values
        stop_loss_pct = self.config.get('stop_loss', 0.01)
        take_profit_pct = self.config.get('take_profit', 0.02)
        
        # Check TP/SL using high/low (intra-candle) for more realistic backtesting
        # For LONG: check if high reached TP or low reached SL
        # For SHORT: check if low reached TP or high reached SL
        if position.side == "BUY":
            # LONG position
            tp_price = entry_price * (1 + take_profit_pct)
            sl_price = entry_price * (1 - stop_loss_pct)
            
            # Check if TP was hit (high reached TP level)
            if candle_high >= tp_price:
                return "TAKE_PROFIT"
            # Check if SL was hit (low reached SL level)
            if candle_low <= sl_price:
                return "STOP_LOSS"
            
            # Calculate P&L at close for time limit checks
            pnl_pct = (candle_close - entry_price) / entry_price
        else:  # SELL
            # SHORT position
            tp_price = entry_price * (1 - take_profit_pct)
            sl_price = entry_price * (1 + stop_loss_pct)
            
            # Check if TP was hit (low reached TP level)
            if candle_low <= tp_price:
                return "TAKE_PROFIT"
            # Check if SL was hit (high reached SL level)
            if candle_high >= sl_price:
                return "STOP_LOSS"
            
            # Calculate P&L at close for time limit checks
            pnl_pct = (entry_price - candle_close) / entry_price
        
        # Time limit: if profitable when it expires and extension is enabled, keep extending while profitable
        time_elapsed = df['timestamp'].iloc[idx] - position.entry_time
        time_limit = self.config.get('time_limit', 3600)
        enable_time_limit_extension = self.config.get('enable_time_limit_extension', False)
        time_limit_extension = self.config.get('time_limit_extension', 300)  # Default 5 minutes
        
        if enable_time_limit_extension:
            # Initialize effective_time_limit on first check
            if position.effective_time_limit is None:
                position.effective_time_limit = time_limit
            
            # Check if we've reached the effective time limit
            if time_elapsed >= position.effective_time_limit:
                # If profitable, keep extending the time limit (no limit on extensions)
                if pnl_pct > 0:
                    position.effective_time_limit += time_limit_extension
                    # Track that we extended (for reporting)
                    if not position.time_limit_extended:
                        position.time_limit_extended = True
                    # Don't close yet, let it continue
                    return None
                else:
                    # Not profitable, close at time limit
                    return "TIME_LIMIT"
        else:
            # Simple time limit - close when time elapsed exceeds limit
            if time_elapsed >= time_limit:
                return "TIME_LIMIT"
        
        return None
    
    async def run_backtest(self, start_time: int, end_time: int):
        """Run the backtest on historical data."""
        # Load historical candles
        await self.load_historical_candles(start_time, end_time)
        
        if self.candles_df is None or len(self.candles_df) == 0:
            logger.error("No candles loaded")
            return
        
        # Calculate RSI
        self.candles_df = self.calculate_rsi(self.candles_df)
        
        # Iterate through candles (no signal sustain - check only at candle close)
        for idx in range(self.config['rsi_lookback_candles'] + 1, len(self.candles_df)):
            timestamp = self.candles_df['timestamp'].iloc[idx]
            price = self.candles_df['close'].iloc[idx]
            
            # Check exit conditions for open position
            if self.current_position and self.current_position.is_open:
                close_reason = self.check_exit_conditions(self.candles_df, idx, self.current_position)
                if close_reason:
                    self.close_position(idx, close_reason)
            
            # Check for new signal (only at candle close, no sustain check)
            signal = self.get_signal(self.candles_df, idx)
            
            if signal != 0:
                # Check cooldown period
                cooldown = self.config.get('cooldown_after_execution', 0)
                can_trade = True
                if cooldown > 0 and self.last_position_close_time is not None:
                    time_since_close = timestamp - self.last_position_close_time
                    if time_since_close < cooldown:
                        can_trade = False
                
                # Check max executors (simplified - just check if position is open)
                max_executors = self.config.get('max_executors', 1)
                has_open_position = self.current_position is not None and self.current_position.is_open
                
                # Check filters and conditions
                if (can_trade and 
                    self.check_filters(self.candles_df, idx, signal) and 
                    not has_open_position):
                    # Open new position
                    self.open_position(idx, signal, price)
        
        # Close any remaining open positions
        if self.current_position and self.current_position.is_open:
            self.close_position(len(self.candles_df) - 1, "BACKTEST_END")
    
    def open_position(self, idx: int, signal: int, price: float):
        """Open a new position."""
        side = "BUY" if signal == 1 else "SELL"
        amount = self.config['order_amount_usd'] / price
        
        position = BacktestResult(
            entry_time=self.candles_df['timestamp'].iloc[idx],
            entry_price=price,
            side=side,
            amount=amount
        )
        
        self.current_position = position
        logger.info(f"Opened {side} position at {price:.2f} (idx {idx})")
    
    def close_position(self, idx: int, close_type: str):
        """Close the current position."""
        if not self.current_position or not self.current_position.is_open:
            return
        
        position = self.current_position
        close_timestamp = self.candles_df['timestamp'].iloc[idx]
        entry_price = position.entry_price
        
        # Use static TP/SL for exit price calculation
        stop_loss_pct = self.config.get('stop_loss', 0.01)
        take_profit_pct = self.config.get('take_profit', 0.02)
        
        # Determine exit price based on close type
        # If TP/SL was hit, use the exact TP/SL price (more realistic)
        # For TIME_LIMIT, use candle close but cap at stop loss to prevent losses exceeding SL
        if close_type == "TAKE_PROFIT":
            if position.side == "BUY":
                exit_price = entry_price * (1 + take_profit_pct)
            else:  # SELL
                exit_price = entry_price * (1 - take_profit_pct)
        elif close_type == "STOP_LOSS":
            if position.side == "BUY":
                exit_price = entry_price * (1 - stop_loss_pct)
            else:  # SELL
                exit_price = entry_price * (1 + stop_loss_pct)
        else:
            # TIME_LIMIT, BACKTEST_END, etc. - use candle close but cap at stop loss
            candle_close = self.candles_df['close'].iloc[idx]
            
            # Calculate stop loss price
            if position.side == "BUY":
                sl_price = entry_price * (1 - stop_loss_pct)
                # For LONG: don't let exit price go below stop loss
                exit_price = max(candle_close, sl_price)
            else:  # SELL
                sl_price = entry_price * (1 + stop_loss_pct)
                # For SHORT: don't let exit price go above stop loss
                exit_price = min(candle_close, sl_price)
        
        position.exit_time = close_timestamp
        position.exit_price = exit_price
        position.close_type = close_type
        position.is_open = False
        
        # Calculate P&L (before fees)
        if position.side == "BUY":
            gross_pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:  # SELL
            gross_pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        # Apply trading fees (entry + exit)
        # Default to 0.06% per trade (0.12% total for round trip) - typical for perpetuals
        # This can be configured in the config
        fee_per_trade = self.config.get('trading_fee_per_trade', 0.0006)  # 0.06% per trade
        total_fee_pct = fee_per_trade * 2  # Entry + Exit
        
        # Net P&L after fees
        position.pnl_pct = gross_pnl_pct - total_fee_pct
        position.pnl_quote = position.pnl_pct * self.config['order_amount_usd']
        
        # Track fees separately for reporting
        position.fees_paid = total_fee_pct * self.config['order_amount_usd']
        
        # Track when position closed for cooldown
        self.last_position_close_time = close_timestamp
        
        self.trades.append(position)
        logger.info(f"Closed {position.side} position: {close_type}, P&L: {position.pnl_pct:.2%}")
        
        self.current_position = None
    
    def print_results(self):
        """Print backtest results."""
        if not self.trades:
            logger.info("No trades executed")
            return
        
        df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'side': t.side,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl_pct': t.pnl_pct,
            'pnl_quote': t.pnl_quote,
            'fees_paid': t.fees_paid,
            'close_type': t.close_type,
        } for t in self.trades])
        
        total_trades = len(df)
        winning = len(df[df['pnl_pct'] > 0])
        losing = len(df[df['pnl_pct'] < 0])
        win_rate_pct = (winning / total_trades * 100) if total_trades > 0 else 0
        win_rate_decimal = win_rate_pct / 100  # For percentage formatting
        
        total_pnl = df['pnl_quote'].sum()
        avg_pnl = df['pnl_pct'].mean()
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(f"Total Trades: {total_trades}")
        print(f"Winning: {winning} ({winning/total_trades*100:.1f}%)")
        print(f"Losing: {losing} ({losing/total_trades*100:.1f}%)")
        print(f"Win Rate: {win_rate_pct:.2f}%")
        total_fees = df['fees_paid'].sum()
        net_pnl = total_pnl - total_fees
        
        print(f"Total P&L (gross): {total_pnl:.2f} USD ({df['pnl_pct'].sum():.2%})")
        print(f"Total Fees: {total_fees:.2f} USD")
        print(f"Total P&L (net): {net_pnl:.2f} USD ({net_pnl/self.config['order_amount_usd']:.2%})")
        print(f"Average P&L: {avg_pnl:.2%}")
        print(f"Largest Win: {df['pnl_pct'].max():.2%}")
        print(f"Largest Loss: {df['pnl_pct'].min():.2%}")
        
        # Risk-reward analysis
        if losing > 0 and winning > 0:
            avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean()
            avg_loss = abs(df[df['pnl_pct'] < 0]['pnl_pct'].mean())
            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            break_even_win_rate = 1 / (1 + risk_reward_ratio) if risk_reward_ratio > 0 else 1
            print(f"\nRisk-Reward Analysis:")
            print(f"Average Win: {avg_win:.2%}")
            print(f"Average Loss: {avg_loss:.2%}")
            print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}:1")
            print(f"Break-even Win Rate Needed: {break_even_win_rate:.1%}")
            print(f"Current Win Rate: {win_rate_decimal:.1%}")
            if win_rate_decimal < break_even_win_rate:
                required_rr = (1 - win_rate_decimal) / win_rate_decimal
                print(f"Win rate too low! Need R:R of {required_rr:.2f}:1 to break even")
                print(f"   Suggested take profit: {avg_loss * required_rr:.4f} ({avg_loss * required_rr * 100:.2f}%)")
        print("\nClose Type Breakdown:")
        print(df['close_type'].value_counts())
        
        # P&L analysis by close type
        print("\nP&L by Close Type:")
        for close_type in df['close_type'].unique():
            subset = df[df['close_type'] == close_type]
            count = len(subset)
            avg_pnl = subset['pnl_pct'].mean()
            total_pnl = subset['pnl_quote'].sum()
            winning = len(subset[subset['pnl_pct'] > 0])
            win_rate = (winning / count * 100) if count > 0 else 0
            print(f"  {close_type}: {count} trades, Avg P&L: {avg_pnl:.2%}, Total: {total_pnl:.2f} USD, Win Rate: {win_rate:.1f}%")
        
        # RSI statistics
        if self.candles_df is not None:
            rsi_column = f"RSI_{self.config['rsi_length']}"
            if rsi_column in self.candles_df.columns:
                lookback = self.config['rsi_lookback_candles']
                # Calculate RSI diff for each candle (current - lookback candles ago)
                rsi_diff = self.candles_df[rsi_column] - self.candles_df[rsi_column].shift(lookback)
                max_rsi_rise = rsi_diff.max()
                max_rsi_drop = rsi_diff.min()
                avg_rsi_diff = rsi_diff.abs().mean()
                print(f"\nRSI Statistics (lookback={lookback} candles):")
                print(f"  Max RSI rise (single candle diff): +{max_rsi_rise:.2f}")
                print(f"  Max RSI drop (single candle diff): {max_rsi_drop:.2f}")
                print(f"  Avg absolute RSI diff: {avg_rsi_diff:.2f}")
                print(f"  RSI direction threshold: {self.config['rsi_direction_threshold']:.2f}")
        
        print("="*80)
        
        # Save to CSV
        csv_path = data_path() + f"/backtest_rsi_trend_follower_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")


async def main():
    """Main backtest function."""
    import yaml
    import sys
    from pathlib import Path
    from hummingbot.client.settings import SCRIPT_STRATEGY_CONF_DIR_PATH
    
    # Load configuration from YAML file - allow command line argument for different configs
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "conf_rsi_trend_follower.yml"
    
    config_path = SCRIPT_STRATEGY_CONF_DIR_PATH / config_file
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        config = {}
    else:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config = yaml_config if yaml_config else {}
            logger.info(f"Loaded config from {config_path}")
    
    # Extract backtest period settings first (before creating config_dict)
    backtest_days = config.get('backtest_days', 7)
    backtest_end_time = config.get('backtest_end_time')
    backtest_start_time = config.get('backtest_start_time')
    
    # Convert YAML values to appropriate types and set defaults
    # Handle candles_exchange and candles_pair (for data source) vs exchange and trading_pair (for trading)
    candles_exchange = config.get('candles_exchange') or config.get('exchange') or 'binance_perpetual'
    candles_pair = config.get('candles_pair') or config.get('trading_pair') or 'ETH-USDT'
    
    # If using Binance and pair is ETH-USD (Hyperliquid format), convert to ETH-USDT
    if 'binance' in candles_exchange.lower() and candles_pair == 'ETH-USD':
        candles_pair = 'ETH-USDT'
        logger.info(f"Converted trading pair from ETH-USD to ETH-USDT for Binance")
    
    config_dict = {
        'candles_exchange': candles_exchange,
        'trading_pair': candles_pair,
        'candles_interval': config.get('candles_interval', '5m'),
        'rsi_length': int(config.get('rsi_length', 5)),
        'rsi_direction_threshold': float(config.get('rsi_direction_threshold', 11.0)),
        'rsi_lookback_candles': int(config.get('rsi_lookback_candles', 2)),
        'rsi_overbought': float(config.get('rsi_overbought', 77.0)),
        'rsi_oversold': float(config.get('rsi_oversold', 24.0)),
        'extreme_rsi_overbought': float(config.get('extreme_rsi_overbought', 85.0)),  # SHORT if RSI >= this (extreme overbought)
        'extreme_rsi_oversold': float(config.get('extreme_rsi_oversold', 15.0)),  # LONG if RSI <= this (extreme oversold)
        'enable_extreme_rsi_signals': bool(config.get('enable_extreme_rsi_signals', True)),  # Enable extreme RSI signals
        'stop_loss': float(config.get('stop_loss', 0.0032)),
        'take_profit': float(config.get('take_profit', 0.008)),
        'time_limit': int(config.get('time_limit', 2400)),
        'enable_time_limit_extension': bool(config.get('enable_time_limit_extension', False)),  # Enable extending time limit for profitable trades
        'time_limit_extension': float(config.get('time_limit_extension', 300)),  # Seconds to extend if profitable (default: 5 minutes)
        'order_amount_usd': float(config.get('order_amount_usd', 500)),
        'cooldown_after_execution': int(config.get('cooldown_after_execution', 300)),
        'max_executors': int(config.get('max_executors', 1)),
        'enable_volatility_filter': bool(config.get('enable_volatility_filter', True)),
        'atr_length': int(config.get('atr_length', 14)),
        'min_volatility_pct': float(config.get('min_volatility_pct', 0.15)),
        'enable_market_condition_filter': bool(config.get('enable_market_condition_filter', True)),
        'adx_length': int(config.get('adx_length', 14)),
        'min_adx_trend_strength': float(config.get('min_adx_trend_strength', 25.0)),  # Minimum ADX (both trend-following and counter-trend need trending markets)
        'rsi_variance_lookback': int(config.get('rsi_variance_lookback', 10)),
        'max_rsi_variance': float(config.get('max_rsi_variance', 400.0)),
        'trading_fee_per_trade': float(config.get('trading_fee_per_trade', 0.0006)),  # 0.06% per trade (default for perpetuals)
        'time_limit_extension': float(config.get('time_limit_extension', 300)),  # Extend time limit by this many seconds if profitable when it expires (default: 300 = 5 minutes)
        'counter_trend_mode': bool(config.get('counter_trend_mode', False)),  # If True, flip signals (LONG becomes SHORT, SHORT becomes LONG) and adjust filters accordingly
    }
    
    # Use config_dict as config
    config = config_dict
    
    # Backtest period - can be configured in YAML or use defaults
    # Option 1: Specify number of days to look back
    # Option 2: Specify exact start/end timestamps (overrides backtest_days if provided)
    end_time = backtest_end_time
    start_time = backtest_start_time
    
    if end_time is None:
        end_time = int(datetime.now().timestamp())
    
    if start_time is None:
        # Use backtest_days to calculate start time
        start_time = int((datetime.fromtimestamp(end_time) - timedelta(days=backtest_days)).timestamp())
    else:
        # If start_time is provided as string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS), convert it
        if isinstance(start_time, str):
            try:
                start_time = int(datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp())
            except:
                start_time = int(datetime.strptime(start_time, '%Y-%m-%d').timestamp())
        elif isinstance(start_time, (int, float)):
            start_time = int(start_time)
    
    # If end_time is provided as string, convert it
    if isinstance(end_time, str):
        try:
            end_time = int(datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp())
        except:
            end_time = int(datetime.strptime(end_time, '%Y-%m-%d').timestamp())
    elif isinstance(end_time, (int, float)):
        end_time = int(end_time)
    
    logger.info(f"Backtest period: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
    logger.info(f"Duration: {(end_time - start_time) / 86400:.1f} days")
    
    backtest = RSITrendFollowerBacktest(config)
    await backtest.run_backtest(start_time, end_time)
    backtest.print_results()


if __name__ == "__main__":
    asyncio.run(main())
