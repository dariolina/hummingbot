import os
from decimal import Decimal
from typing import Dict, Optional

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor


class RSITrendFollowerConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    exchange: str = Field(default="hyperliquid_perpetual")
    trading_pair: str = Field(default="ETH-USD")
    order_amount_usd: Decimal = Field(default=Decimal("40"), gt=0)
    leverage: int = Field(default=10, gt=0)
    max_executors: int = Field(default=1, gt=0, description="Maximum number of concurrent position executors")
    position_mode: PositionMode = Field(default=PositionMode.ONEWAY)
    cooldown_after_execution: int = Field(default=30, ge=0, description="Cooldown period in seconds before opening a new position after the last executor")
    
    # Position parameters
    stop_loss: float = Field(default=0.01, gt=0, description="Stop loss percentage")
    take_profit: float = Field(default=0.02, gt=0, description="Take profit percentage")
    time_limit: int = Field(default=3600, gt=0)
    enable_time_limit_extension: bool = Field(default=False, description="If True, extend time limit for profitable positions when it expires")
    time_limit_extension: int = Field(default=300, ge=0, description="Seconds to extend time limit if profitable when it expires")
    trailing_stop_activation_delta: float = Field(default=0.02, gt=0, description="Trailing stop activation delta (set equal to take_profit to avoid early activation)")
    trailing_stop_trailing_delta: float = Field(default=0.001, gt=0, description="Trailing stop trailing delta")
    
    # RSI parameters
    rsi_length: int = Field(default=14, gt=0)
    rsi_direction_threshold: float = Field(default=3.0, gt=0)
    rsi_lookback_candles: int = Field(default=1, gt=0, description="Number of candles back to compare RSI (1 = previous candle, higher = catch rapid changes)")
    rsi_overbought: float = Field(default=70.0, ge=0, le=100, description="RSI overbought threshold (skip LONG if RSI > this)")
    rsi_oversold: float = Field(default=30.0, ge=0, le=100, description="RSI oversold threshold (skip SHORT if RSI < this)")
    candles_interval: str = Field(default="5m")
    candles_exchange: Optional[str] = Field(default=None, description="Exchange for candles (defaults to trading exchange)")
    
    # Counter-trend mode
    counter_trend_mode: bool = Field(default=False, description="If True, flip signals (LONG becomes SHORT, SHORT becomes LONG) for mean reversion")
    
    # Extreme RSI signals (high probability reversals)
    enable_extreme_rsi_signals: bool = Field(default=True, description="Enable extreme RSI signals that bypass threshold filters")
    extreme_rsi_overbought: float = Field(default=85.0, ge=0, le=100, description="SHORT if RSI >= this (extreme overbought)")
    extreme_rsi_oversold: float = Field(default=15.0, ge=0, le=100, description="LONG if RSI <= this (extreme oversold)")
    
    # Volatility filter parameters
    enable_volatility_filter: bool = Field(default=True, description="Enable volatility filter to avoid choppy markets")
    atr_length: int = Field(default=14, gt=0, description="ATR period length in candles")
    min_volatility_pct: float = Field(default=0.15, ge=0, description="Minimum NATR percentage to enter position (e.g., 0.15 = 0.15%)")
    
    # Market condition detection parameters
    enable_market_condition_filter: bool = Field(default=True, description="Enable market condition detection to avoid choppy/range-bound markets")
    adx_length: int = Field(default=14, gt=0, description="ADX period length for trend strength detection")
    min_adx_trend_strength: float = Field(default=25.0, ge=0, description="Minimum ADX value to consider market trending (lower = choppy)")
    rsi_variance_lookback: int = Field(default=10, gt=0, description="Number of candles to calculate RSI variance for choppiness detection")
    max_rsi_variance: float = Field(default=400.0, ge=0, description="Maximum RSI variance to allow trading (higher = more choppy allowed)")
    
    # Order types
    open_order_type: OrderType = Field(default=OrderType.MARKET)
    take_profit_order_type: OrderType = Field(default=OrderType.MARKET)
    stop_loss_order_type: OrderType = Field(default=OrderType.MARKET)
    time_limit_order_type: OrderType = Field(default=OrderType.MARKET)
    
    @field_validator('open_order_type', 'take_profit_order_type', 'stop_loss_order_type', 'time_limit_order_type', mode="before")
    @classmethod
    def validate_order_type(cls, v):
        if isinstance(v, int):
            return OrderType(v)
        if isinstance(v, str):
            return OrderType(int(v))
        return v
    
    @field_validator('position_mode', mode="before")
    @classmethod
    def validate_position_mode(cls, v):
        if isinstance(v, str):
            if v.upper() in PositionMode.__members__:
                return PositionMode[v.upper()]
            raise ValueError(f"Invalid position mode: {v}. Valid options: {', '.join(PositionMode.__members__)}")
        return v


class RSITrendFollower(DirectionalStrategyBase):
    """
    Simple RSI trend-following strategy.
    
    This strategy uses RSI direction to generate trading signals:
    - Trend-following mode: LONG when RSI rising, SHORT when RSI dropping
    - Counter-trend mode: SHORT when RSI rising (expecting reversal), LONG when RSI dropping
    
    Also supports extreme RSI signals that bypass threshold requirements.
    
    Parameters:
        directional_strategy_name (str): The name of the strategy.
        trading_pair (str): The trading pair to be traded.
        exchange (str): The exchange to be used for trading.
        order_amount_usd (Decimal): The amount of the order in USD.
        leverage (int): The leverage to be used for trading.
    
    Position Parameters:
        stop_loss (float): The stop-loss percentage for the position.
        take_profit (float): The take-profit percentage for the position.
        time_limit (int): The time limit for the position in seconds.
    
    Candlestick Configuration:
        candles (List[CandlesBase]): The list of candlesticks used for generating signals.
    
    Markets:
        A dictionary specifying the markets and trading pairs for the strategy.
    
    Methods:
        get_signal(): Generates the trading signal based on RSI direction.
        get_processed_df(): Retrieves the processed dataframe with RSI values.
        market_data_extra_info(): Provides additional information about the market data.
    
    Inherits from:
        DirectionalStrategyBase: Base class for creating directional strategies using the PositionExecutor.
    """
    directional_strategy_name: str = "RSI_trend_follower"
    
    markets: Dict[str, set] = {}
    
    @classmethod
    def init_markets(cls, config: RSITrendFollowerConfig):
        """Initialize markets from config."""
        cls.markets = {config.exchange: {config.trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: Optional[RSITrendFollowerConfig] = None):
        # Use provided config or create default
        if config is None:
            config = RSITrendFollowerConfig()
        self.config = config
        
        # Set instance attributes from config
        self.trading_pair = config.trading_pair
        self.exchange = config.exchange
        self.order_amount_usd = config.order_amount_usd
        self.leverage = config.leverage
        self.max_executors = config.max_executors
        self.position_mode = config.position_mode
        self.cooldown_after_execution = config.cooldown_after_execution
        self.stop_loss = config.stop_loss
        self.take_profit = config.take_profit
        self.time_limit = config.time_limit
        self.rsi_length = config.rsi_length
        self.rsi_direction_threshold = config.rsi_direction_threshold
        self.rsi_lookback_candles = config.rsi_lookback_candles
        self.rsi_overbought = config.rsi_overbought
        self.rsi_oversold = config.rsi_oversold
        self.enable_volatility_filter = config.enable_volatility_filter
        self.atr_length = config.atr_length
        self.min_volatility_pct = config.min_volatility_pct
        self.enable_market_condition_filter = config.enable_market_condition_filter
        self.adx_length = config.adx_length
        self.min_adx_trend_strength = config.min_adx_trend_strength
        self.rsi_variance_lookback = config.rsi_variance_lookback
        self.max_rsi_variance = config.max_rsi_variance
        
        # Counter-trend mode and extreme RSI
        self.counter_trend_mode = config.counter_trend_mode
        self.enable_extreme_rsi_signals = config.enable_extreme_rsi_signals
        self.extreme_rsi_overbought = config.extreme_rsi_overbought
        self.extreme_rsi_oversold = config.extreme_rsi_oversold
        
        # Time limit extension
        self.enable_time_limit_extension = config.enable_time_limit_extension
        self.time_limit_extension = config.time_limit_extension
        
        # Trailing stop
        self.trailing_stop_activation_delta = config.trailing_stop_activation_delta
        self.trailing_stop_trailing_delta = config.trailing_stop_trailing_delta
        
        # Track time limit extensions per executor
        self._time_limit_extended: Dict[str, bool] = {}
        
        self.candles_interval = config.candles_interval
        self.candles_exchange = config.candles_exchange or self.exchange
        self.open_order_type = config.open_order_type
        self.take_profit_order_type = config.take_profit_order_type
        self.stop_loss_order_type = config.stop_loss_order_type
        self.time_limit_order_type = config.time_limit_order_type
        
        # Initialize markets before parent init
        self.markets = {self.exchange: {self.trading_pair}}
        
        # Initialize candles
        self.candles = [
            CandlesFactory.get_candle(
                CandlesConfig(
                    connector=self.candles_exchange,
                    trading_pair=self.trading_pair,
                    interval=self.candles_interval,
                    max_records=1000
                )
            )
        ]
        
        # Call parent init (will create triple_barrier_conf and start candles)
        super().__init__(connectors)
    
    def on_tick(self):
        """
        Main tick handler - checks conditions and creates positions.
        """
        self.clean_and_store_executors()
        if self.is_perpetual:
            self.check_and_set_leverage()
        
        # Check conditions
        signal = self.get_signal()
        max_executors_ok = self.max_active_executors_condition
        candles_ready = self.all_candles_ready
        cooldown_ok = self.time_between_signals_condition
        
        if signal != 0:
            if not max_executors_ok:
                self.logger().debug(
                    f"Signal {signal} detected but blocked: max_executors condition "
                    f"({len(self.get_active_executors())} >= {self.max_executors})"
                )
            elif not candles_ready:
                self.logger().debug(
                    f"Signal {signal} detected but blocked: candles not ready"
                )
            elif not cooldown_ok:
                seconds_since_last = self.current_timestamp - self.get_timestamp_of_last_executor()
                self.logger().debug(
                    f"Signal {signal} detected but blocked: cooldown ({seconds_since_last:.1f}s < {self.cooldown_after_execution}s)"
                )
        
        # Check time limit extension for profitable positions
        if self.enable_time_limit_extension and self.get_active_executors():
            for executor in self.get_active_executors():
                self.check_time_limit_extension(executor)
        
        if max_executors_ok and candles_ready and cooldown_ok:
            position_config = self.get_position_config()
            if position_config:
                signal_executor = PositionExecutor(
                    strategy=self,
                    config=position_config,
                )
                signal_executor.start()
                self.active_executors.append(signal_executor)
                self.logger().info(
                    f"Created and started position executor {signal_executor.config.id} "
                    f"(signal: {signal}, side: {position_config.side.name})"
                )

    def get_signal(self):
        """
        Generates the trading signal based on RSI direction.
        Compares current RSI to RSI from N candles ago to catch rapid changes.
        
        Supports two modes:
        - Trend-following (default): LONG when RSI rising, SHORT when RSI dropping
        - Counter-trend (counter_trend_mode=True): SHORT when RSI rising (expecting reversal), LONG when RSI dropping
        
        Also supports extreme RSI signals that bypass threshold requirements.
        
        Returns:
            int: The trading signal (-1 for sell/short, 1 for buy/long, 0 for hold).
        """
        candles_df = self.get_processed_df()
        
        # Need at least lookback_candles + 1 candles
        if len(candles_df) < self.rsi_lookback_candles + 1:
            return 0
        
        rsi_column = f"RSI_{self.rsi_length}"
        if rsi_column not in candles_df.columns:
            return 0
        
        current_rsi = candles_df[rsi_column].iloc[-1]
        # Compare to RSI from N candles ago (e.g., 2-3 candles back catches rapid drops)
        lookback_rsi = candles_df[rsi_column].iloc[-(self.rsi_lookback_candles+1)]
        # Also get previous candle for trend-following sanity check
        previous_rsi = candles_df[rsi_column].iloc[-2] if len(candles_df) >= 2 else current_rsi
        
        signal = 0
        
        # Extreme RSI signals - high probability reversals regardless of trend
        # These bypass the normal threshold requirement
        if self.enable_extreme_rsi_signals:
            # Extreme overbought → SHORT (expecting reversal down)
            if current_rsi >= self.extreme_rsi_overbought:
                self.logger().info(
                    f"EXTREME RSI overbought: {current_rsi:.2f} >= {self.extreme_rsi_overbought:.2f} → SHORT signal"
                )
                return -1
            # Extreme oversold → LONG (expecting reversal up)
            if current_rsi <= self.extreme_rsi_oversold:
                self.logger().info(
                    f"EXTREME RSI oversold: {current_rsi:.2f} <= {self.extreme_rsi_oversold:.2f} → LONG signal"
                )
                return 1
        
        if self.counter_trend_mode:
            # Counter-trend: Bet against the momentum (mean reversion)
            # SHORT when RSI is rising a lot (expecting reversal down)
            # LONG when RSI is dropping a lot (expecting reversal up)
            if current_rsi >= lookback_rsi + self.rsi_direction_threshold:
                # RSI is rising strongly - bet SHORT (expecting reversal)
                self.logger().info(
                    f"Counter-trend: RSI rising {current_rsi:.2f} >= {lookback_rsi:.2f} + {self.rsi_direction_threshold:.2f} → SHORT"
                )
                signal = -1
            elif current_rsi <= lookback_rsi - self.rsi_direction_threshold:
                # RSI is dropping strongly - bet LONG (expecting reversal)
                self.logger().info(
                    f"Counter-trend: RSI dropping {current_rsi:.2f} <= {lookback_rsi:.2f} - {self.rsi_direction_threshold:.2f} → LONG"
                )
                signal = 1
        else:
            # Trend-following: Enter LONG when RSI is rising, SHORT when RSI is dropping
            # SHORT signal: RSI dropping
            if current_rsi <= lookback_rsi - self.rsi_direction_threshold and current_rsi < previous_rsi:
                self.logger().info(
                    f"RSI is dropping: {current_rsi:.2f} <= {lookback_rsi:.2f} - {self.rsi_direction_threshold:.2f} "
                    f"(vs prev: {previous_rsi:.2f})"
                )
                signal = -1
            # LONG signal: RSI rising
            elif current_rsi >= lookback_rsi + self.rsi_direction_threshold and current_rsi > previous_rsi:
                self.logger().info(
                    f"RSI is rising: {current_rsi:.2f} >= {lookback_rsi:.2f} + {self.rsi_direction_threshold:.2f} "
                    f"(vs prev: {previous_rsi:.2f})"
                )
                signal = 1
        
        if signal == 0:
            diff = current_rsi - lookback_rsi
            self.logger().info(f"RSI normal: diff {diff:.2f} (current: {current_rsi:.2f}, lookback: {lookback_rsi:.2f}, prev: {previous_rsi:.2f})")
        
        return signal

    def check_time_limit_extension(self, executor: PositionExecutor) -> bool:
        """
        Check if time limit should be extended for a profitable position.
        
        If the position is profitable when the time limit is approaching,
        extend the time limit by time_limit_extension seconds (one time only).
        
        Args:
            executor: The position executor to check
            
        Returns:
            True if time limit was extended, False otherwise
        """
        try:
            from hummingbot.strategy_v2.models.base import RunnableStatus
            
            executor_id = executor.config.id
            
            # Skip if executor is already closed or shutting down
            if executor.is_closed or executor.status == RunnableStatus.SHUTTING_DOWN:
                # Clean up tracking
                self._time_limit_extended.pop(executor_id, None)
                return False
            
            # Skip if already extended for this executor
            if self._time_limit_extended.get(executor_id, False):
                return False
            
            # Skip if no time limit configured
            if executor.config.triple_barrier_config.time_limit is None:
                return False
            
            # Check if approaching time limit (within 5 seconds)
            time_elapsed = self.current_timestamp - executor.config.timestamp
            time_limit = executor.config.triple_barrier_config.time_limit
            
            if time_elapsed >= time_limit - 5:  # Within 5 seconds of time limit
                # Check if position is profitable
                pnl_pct = executor.get_net_pnl_pct()
                
                if pnl_pct > Decimal("0"):
                    # Extend time limit
                    new_time_limit = time_limit + self.time_limit_extension
                    executor.config.triple_barrier_config.time_limit = new_time_limit
                    self._time_limit_extended[executor_id] = True
                    
                    self.logger().info(
                        f"Executor {executor_id}: Extended time limit by {self.time_limit_extension}s "
                        f"(PNL: {pnl_pct:.4%}, new time limit: {new_time_limit}s)"
                    )
                    return True
            
            return False
        except Exception as e:
            self.logger().debug(f"Error checking time limit extension for executor {executor.config.id}: {e}")
            return False
    
    def is_extreme_rsi_signal(self, current_rsi: float) -> bool:
        """Check if current RSI is at extreme levels (bypasses other filters)."""
        if not self.enable_extreme_rsi_signals:
            return False
        return current_rsi >= self.extreme_rsi_overbought or current_rsi <= self.extreme_rsi_oversold
    
    def get_position_config(self):
        """
        Override to add RSI threshold filtering and NATR volatility filter before creating position.
        Supports counter-trend mode and extreme RSI signals.
        """
        signal = self.get_signal()
        if signal == 0:
            return None
        
        # Check RSI thresholds and NATR volatility filter before creating position
        rsi_data = self.get_rsi_data(include_natr=self.enable_volatility_filter, 
                                      include_market_condition=self.enable_market_condition_filter)
        if rsi_data is not None:
            try:
                current_rsi = rsi_data["current_rsi"]
                
                # Check if this is an extreme RSI signal (bypasses RSI threshold and market condition filters, but NOT volatility)
                is_extreme = self.is_extreme_rsi_signal(current_rsi)
                
                if not is_extreme:
                    # Non-extreme signals need to pass RSI threshold and market condition filters
                    
                    # Market condition filter: Skip if market is choppy/range-bound
                    if self.enable_market_condition_filter and "market_condition" in rsi_data:
                        market_condition = rsi_data["market_condition"]
                        if market_condition["is_choppy"]:
                            self.logger().info(
                                f"Skipping signal: Market is choppy/range-bound "
                                f"(ADX: {market_condition['adx']:.2f} < {self.min_adx_trend_strength}, "
                                f"RSI variance: {market_condition['rsi_variance']:.2f} > {self.max_rsi_variance}, "
                                f"RSI: {current_rsi:.2f})"
                            )
                            return None
                        else:
                            self.logger().debug(
                                f"Market condition filter passed: {market_condition['condition']} "
                                f"(ADX: {market_condition['adx']:.2f}, RSI variance: {market_condition['rsi_variance']:.2f})"
                            )
                    
                    # RSI threshold filters - adjust based on counter-trend mode
                    if self.counter_trend_mode:
                        # Counter-trend: SHORT when RSI is rising (expecting reversal), LONG when RSI is dropping (expecting reversal)
                        # We WANT to SHORT when RSI is high (overbought) and LONG when RSI is low (oversold)
                        # So we filter SHORT when RSI is too low, LONG when RSI is too high
                        if signal == -1 and current_rsi < self.rsi_oversold:
                            self.logger().info(
                                f"Skipping SHORT signal: RSI {current_rsi:.2f} < {self.rsi_oversold} "
                                f"(too low for counter-trend SHORT)"
                            )
                            return None
                        if signal == 1 and current_rsi > self.rsi_overbought:
                            self.logger().info(
                                f"Skipping LONG signal: RSI {current_rsi:.2f} > {self.rsi_overbought} "
                                f"(too high for counter-trend LONG)"
                            )
                            return None
                    else:
                        # Trend-following: LONG when RSI rising, SHORT when RSI dropping
                        # Filter LONG when overbought, SHORT when oversold
                        if signal == 1 and current_rsi > self.rsi_overbought:
                            self.logger().info(
                                f"Skipping LONG signal: RSI {current_rsi:.2f} > {self.rsi_overbought} (overbought)"
                            )
                            return None
                        
                        if signal == -1 and current_rsi < self.rsi_oversold:
                            self.logger().info(
                                f"Skipping SHORT signal: RSI {current_rsi:.2f} < {self.rsi_oversold} (oversold)"
                            )
                            return None
                
                # Volatility filter: Skip if volatility is too low (choppy market)
                # Applies to ALL signals, including extreme RSI
                if self.enable_volatility_filter:
                    if "natr" in rsi_data:
                        natr_value = rsi_data["natr"]
                        if natr_value < self.min_volatility_pct:
                            self.logger().info(
                                f"Skipping signal: NATR {natr_value:.3f}% < {self.min_volatility_pct}% "
                                f"(choppy market, RSI: {current_rsi:.2f})"
                            )
                            return None
                        else:
                            self.logger().debug(
                                f"Volatility filter passed: NATR {natr_value:.3f}% >= {self.min_volatility_pct}% "
                                f"(RSI: {current_rsi:.2f})"
                            )
                    else:
                        self.logger().warning(
                            f"Volatility filter enabled but NATR not available in rsi_data. "
                            f"Keys: {list(rsi_data.keys())}"
                        )
            except Exception as e:
                self.logger().error(f"Error checking RSI thresholds/NATR: {e}", exc_info=True)
        
        # Get price and calculate position amount
        price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
        side = TradeType.BUY if signal == 1 else TradeType.SELL
        if self.open_order_type.is_limit_type():
            price = price * (1 - signal * self.open_order_slippage_buffer)
        
        from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
        
        position_config = PositionExecutorConfig(
            timestamp=self.current_timestamp,
            trading_pair=self.trading_pair,
            connector_name=self.exchange,
            side=side,
            amount=self.order_amount_usd / price,
            entry_price=price,
            triple_barrier_config=self.triple_barrier_conf,
            leverage=self.leverage,
        )
        return position_config

    def get_processed_df(self):
        """
        Retrieves the processed dataframe with RSI values.
        Returns:
            pd.DataFrame: The processed dataframe with RSI values.
        """
        candles_df = self.candles[0].candles_df
        candles_df.ta.rsi(length=self.rsi_length, append=True)
        return candles_df
    
    def get_market_condition(self, candles_df) -> Optional[Dict]:
        """
        Detect market condition (trending vs choppy/range-bound).
        
        Returns:
            Dict with keys: adx, is_trending, rsi_variance, is_choppy, condition
            Returns None if insufficient data
        """
        try:
            import pandas_ta as ta
            
            # Need enough candles for ADX and RSI variance
            min_candles = max(self.adx_length, self.rsi_variance_lookback) + 5
            if len(candles_df) < min_candles:
                return None
            
            rsi_column = f"RSI_{self.rsi_length}"
            if rsi_column not in candles_df.columns:
                return None
            
            # Calculate ADX for trend strength
            adx = ta.adx(
                candles_df["high"],
                candles_df["low"],
                candles_df["close"],
                length=self.adx_length
            )
            adx_column = f"ADX_{self.adx_length}"
            if adx_column not in adx.columns:
                return None
            
            current_adx = adx[adx_column].iloc[-1]
            is_trending = current_adx >= self.min_adx_trend_strength
            
            # Calculate RSI variance for choppiness detection
            rsi_values = candles_df[rsi_column].tail(self.rsi_variance_lookback)
            rsi_variance = float(rsi_values.var())
            is_choppy_by_rsi = rsi_variance > self.max_rsi_variance
            
            # Overall condition
            is_choppy = not is_trending or is_choppy_by_rsi
            condition = "choppy" if is_choppy else "trending"
            
            return {
                "adx": current_adx,
                "is_trending": is_trending,
                "rsi_variance": rsi_variance,
                "is_choppy_by_rsi": is_choppy_by_rsi,
                "is_choppy": is_choppy,
                "condition": condition,
            }
        except Exception as e:
            self.logger().debug(f"Error calculating market condition: {e}")
            return None
    
    def get_rsi_data(self, include_natr: bool = False, include_market_condition: bool = False) -> Optional[Dict]:
        """
        Calculate and return RSI data for use in position opening/closing decisions.
        
        Args:
            include_natr: If True, also calculate and return NATR value
            include_market_condition: If True, also calculate and return market condition
            
        Returns:
            Dict with keys: current_rsi, lookback_rsi, rsi_change, natr (if include_natr=True), 
            market_condition (if include_market_condition=True)
            Returns None if candles not ready or insufficient data
        """
        if not self.candles or not self.candles[0].ready:
            return None
        
        try:
            candles_df = self.get_processed_df()
            rsi_column = f"RSI_{self.rsi_length}"
            
            if rsi_column not in candles_df.columns or len(candles_df) < self.rsi_lookback_candles + 1:
                return None
            
            current_rsi = candles_df[rsi_column].iloc[-1]
            lookback_rsi = candles_df[rsi_column].iloc[-(self.rsi_lookback_candles+1)]
            previous_rsi = candles_df[rsi_column].iloc[-2] if len(candles_df) >= 2 else current_rsi
            rsi_change = current_rsi - lookback_rsi
            rsi_change_prev = current_rsi - previous_rsi
            
            result = {
                "current_rsi": current_rsi,
                "lookback_rsi": lookback_rsi,
                "previous_rsi": previous_rsi,
                "rsi_change": rsi_change,
                "rsi_change_prev": rsi_change_prev,
            }
            
            # Calculate NATR if requested (for volatility filter)
            if include_natr:
                import pandas_ta as ta
                natr = ta.natr(
                    candles_df["high"],
                    candles_df["low"],
                    candles_df["close"],
                    length=self.atr_length
                )
                if natr is not None and len(natr) > 0:
                    natr_value = natr.iloc[-1]
                    result["natr"] = natr_value
            
            # Calculate market condition if requested
            if include_market_condition:
                market_condition = self.get_market_condition(candles_df)
                if market_condition:
                    result["market_condition"] = market_condition
            
            return result
        except Exception as e:
            self.logger().debug(f"Error calculating RSI data: {e}")
            return None

    def market_data_extra_info(self):
        """
        Provides additional information about the market data to the format status.
        Returns:
            List[str]: A list of formatted strings containing market data information.
        """
        lines = []
        candles_df = self.get_processed_df()
        rsi_column = f"RSI_{self.rsi_length}"
        
        if rsi_column in candles_df.columns and len(candles_df) >= self.rsi_lookback_candles + 1:
            current_rsi = candles_df[rsi_column].iloc[-1]
            lookback_rsi = candles_df[rsi_column].iloc[-(self.rsi_lookback_candles+1)]
            rsi_change = current_rsi - lookback_rsi
            direction = "DROPPING" if rsi_change < 0 else "RISING"
            
            columns_to_show = ["timestamp", "open", "low", "high", "close", "volume", rsi_column]
            lines.extend([f"Candles: {self.candles[0].name} | Interval: {self.candles[0].interval}\n"])
            lines.extend([f"RSI: {current_rsi:.2f} (vs {self.rsi_lookback_candles} candles ago: {lookback_rsi:.2f}, {direction}, change: {rsi_change:+.2f})\n"])
            lines.extend(self.candles_formatted_list(candles_df, columns_to_show))
        else:
            lines.extend([f"Candles: {self.candles[0].name} | Interval: {self.candles[0].interval}\n"])
            lines.extend(["Waiting for RSI data...\n"])
        
        return lines
    
    async def on_stop(self):
        """
        Override to make on_stop async (required by stop command).
        """
        # Call parent on_stop (synchronous method, no await needed)
        super().on_stop()
