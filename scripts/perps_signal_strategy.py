import asyncio
import os
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any

import aiohttp
from pydantic import Field, field_validator

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.base import RunnableStatus


class PerpsSignalStrategyConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    exchange: str = Field(default="hyperliquid_perpetual")
    trading_pair: str = Field(default="BTC-USD")
    api_url: str = Field(default="http://agents.eternax.ai/perps_signal")
    poll_interval: float = Field(default=30.0, gt=0)
    max_executors: int = Field(default=1, gt=0)
    position_mode: PositionMode = Field(default="ONEWAY")
    leverage: int = Field(default=10, gt=0)
    margin_usd: Decimal = Field(default=Decimal("10"), gt=0, description="Margin amount in USD (position value = margin * leverage)")
    stop_loss: float = Field(default=0.03, gt=0)
    take_profit: float = Field(default=0.01, gt=0)
    time_limit: int = Field(default=3600, gt=0)
    open_order_type: OrderType = Field(default=OrderType.MARKET)
    take_profit_order_type: OrderType = Field(default=OrderType.MARKET)
    stop_loss_order_type: OrderType = Field(default=OrderType.MARKET)
    time_limit_order_type: OrderType = Field(default=OrderType.MARKET)
    trailing_stop_activation_delta: float = Field(default=0.003, gt=0, description="Trailing stop activation threshold (e.g., 0.003 = 0.3%). Position must reach this PNL% before trailing stop activates.")
    trailing_stop_trailing_delta: float = Field(default=0.001, gt=0, description="Trailing stop delta (e.g., 0.001 = 0.1%). Once activated, stop loss trails by this amount.")
    
    # Technical indicator confirmation parameters
    rsi_length: int = Field(default=14, gt=0, description="RSI period length")
    rsi_overbought: float = Field(default=70.0, ge=0, le=100, description="RSI overbought threshold (skip LONG if RSI > this)")
    rsi_oversold: float = Field(default=30.0, ge=0, le=100, description="RSI oversold threshold (skip SHORT if RSI < this)")
    rsi_extreme_overbought: float = Field(default=85.0, ge=0, le=100, description="RSI extreme overbought threshold (skip LONG if RSI > this)")
    rsi_extreme_oversold: float = Field(default=15.0, ge=0, le=100, description="RSI extreme oversold threshold (skip SHORT if RSI < this)")
    stoch_rsi_length: int = Field(default=5, ge=2, description="Stochastic RSI lookback period")
    stoch_rsi_k: int = Field(default=2, ge=2, description="Stochastic RSI %K smoothing period (min 2)")
    stoch_rsi_d: int = Field(default=2, ge=2, description="Stochastic RSI %D smoothing period (min 2)")
    bb_length: int = Field(default=20, gt=0, description="Bollinger Bands period length")
    bb_std: float = Field(default=2.0, gt=0, description="Bollinger Bands standard deviation")
    bb_upper_threshold: float = Field(default=0.8, ge=0, le=1, description="BBP upper threshold (skip LONG if BBP > this)")
    bb_lower_threshold: float = Field(default=0.2, ge=0, le=1, description="BBP lower threshold (skip SHORT if BBP < this)")
    enable_rsi_override: bool = Field(default=False, description="If RSI contradicts signal, flip signal direction instead of skipping (e.g., SHORT signal + oversold RSI → trade LONG)")
    candles_interval: str = Field(default="5m", description="Candles interval for technical indicators")
    candles_exchange: Optional[str] = Field(default=None, description="Exchange for candles (defaults to trading exchange)")
    small_loss_threshold: float = Field(default=-0.005, le=0, description="Keep unprofitable positions open if PNL > this threshold (e.g., -0.005 = -0.5%)")
    # Volatility filter parameters
    enable_volatility_filter: bool = Field(default=True, description="Enable volatility filter to avoid choppy markets")
    atr_length: int = Field(default=14, gt=0, description="ATR period length in candles (e.g., 14 = 14 candles)")
    min_volatility_pct: float = Field(default=0.15, ge=0, description="Minimum NATR percentage to enter position (e.g., 0.5 = 0.5%)")
    
    @field_validator('position_mode', mode="before")
    @classmethod
    def validate_position_mode(cls, v):
        if isinstance(v, str):
            if v.upper() in PositionMode.__members__:
                return PositionMode[v.upper()]
            raise ValueError(f"Invalid position mode: {v}. Valid options are: {', '.join(PositionMode.__members__)}")
        return v


class PerpsSignalStrategy(DirectionalStrategyBase):
    """
    Directional trading strategy that uses signals from the Perps Signal API.
    
    This strategy polls the API endpoint every 30 seconds and executes positions
    based on the received signals. It tracks market_ids to prevent duplicate
    positions for the same market.
    """
    
    directional_strategy_name: str = "perps_signal"
    
    markets: Dict[str, set] = {}
    
    @classmethod
    def init_markets(cls, config: PerpsSignalStrategyConfig):
        """Initialize markets from config."""
        cls.markets = {config.exchange: {config.trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: Optional[PerpsSignalStrategyConfig] = None):
        # Use provided config or create default
        if config is None:
            config = PerpsSignalStrategyConfig()
        self.config = config
        
        # Set instance attributes from config
        self.trading_pair = config.trading_pair
        self.exchange = config.exchange
        # Log the exchange being used for debugging
        self.logger().info(f"PerpsSignalStrategy initialized with exchange: {self.exchange}")
        self.api_url = config.api_url
        self.poll_interval = config.poll_interval
        self.max_executors = config.max_executors
        self.position_mode = config.position_mode
        self.leverage = config.leverage
        self.margin_usd = config.margin_usd
        self.stop_loss = config.stop_loss
        self.take_profit = config.take_profit
        self.time_limit = config.time_limit
        self.open_order_type = config.open_order_type
        self.take_profit_order_type = config.take_profit_order_type
        self.stop_loss_order_type = config.stop_loss_order_type
        self.time_limit_order_type = config.time_limit_order_type
        self.trailing_stop_activation_delta = config.trailing_stop_activation_delta
        self.trailing_stop_trailing_delta = config.trailing_stop_trailing_delta
        
        # Technical indicator confirmation parameters
        self.rsi_length = config.rsi_length
        self.rsi_overbought = config.rsi_overbought
        self.rsi_oversold = config.rsi_oversold
        self.rsi_extreme_overbought = config.rsi_extreme_overbought
        self.rsi_extreme_oversold = config.rsi_extreme_oversold
        self.stoch_rsi_length = config.stoch_rsi_length
        self.stoch_rsi_k = config.stoch_rsi_k
        self.stoch_rsi_d = config.stoch_rsi_d
        self.bb_length = config.bb_length
        self.bb_std = config.bb_std
        self.bb_upper_threshold = config.bb_upper_threshold
        self.bb_lower_threshold = config.bb_lower_threshold
        self.enable_rsi_override = config.enable_rsi_override
        self.candles_interval = config.candles_interval
        self.candles_exchange = config.candles_exchange or self.exchange
        # Volatility filter parameters
        self.enable_volatility_filter = config.enable_volatility_filter
        self.atr_length = config.atr_length
        self.min_volatility_pct = config.min_volatility_pct
        self.small_loss_threshold = config.small_loss_threshold
        
        # Initialize markets before parent init
        self.markets = {self.exchange: {self.trading_pair}}
        
        # Initialize candles for technical indicator confirmation
        self.candles = [
            CandlesFactory.get_candle(
                CandlesConfig(
                    connector=self.candles_exchange,
                    trading_pair=self.trading_pair,
                    interval=self.candles_interval,
                    max_records=200
                )
            )
        ]
        
        # Initialize market_id tracking
        self._tracked_market_id: Optional[str] = None
        self._traded_market_ids: set = set()  # Track market_ids that have already been traded
        self._executor_market_ids: Dict[str, str] = {}  # Map executor_id -> market_id (or rsi_market_id)
        self._pending_market_id: Optional[str] = None  # Market ID for position being created
        self._last_api_response: Optional[Dict[str, Any]] = None
        self._current_signal: int = 0
        self._api_polling_task: Optional[asyncio.Task] = None
        self._http_client: Optional[aiohttp.ClientSession] = None
        
        # Call parent init (will create triple_barrier_conf and start candles)
        super().__init__(connectors)
        
        # Start API polling task
        self._api_polling_task = safe_ensure_future(self._poll_api_loop())
    
    @property
    def all_candles_ready(self):
        """
        Override to check if candles are ready for technical indicator confirmation.
        """
        if not self.candles or len(self.candles) == 0:
            return True  # If no candles configured, allow trading
        return all(candle.ready for candle in self.candles)
    
    def _http_client_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP client session."""
        if self._http_client is None or self._http_client.closed:
            self._http_client = aiohttp.ClientSession()
        return self._http_client
    
    async def _fetch_signal(self) -> Optional[Dict[str, Any]]:
        """
        Fetch signal from API and return dict with signal and market_id.
        
        Returns:
            Dict with keys: signal (int), market_id (str), bias (float), 
            reason (str), timestamp (str), or None on error
        """
        try:
            client = self._http_client_session()
            timeout = aiohttp.ClientTimeout(total=10)
            async with client.get(self.api_url, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger().error(
                        f"API returned status {response.status}: {error_text}",
                        exc_info=False
                    )
                    return None
                
                data = await response.json()
                
                # Handle server errors (500 status with error field)
                if 'error' in data and data.get('error') == "Failed to generate perps signal":
                    self.logger().error(
                        f"API error: {data.get('details', 'Unknown error')}",
                        exc_info=False
                    )
                    return None
                
                # Map action to signal
                signal = 0
                action = data.get('action', 'no-trade')
                if 'error' not in data and action != 'no-trade':
                    if action == 'long':
                        signal = 1
                    elif action == 'short':
                        signal = -1
                
                return {
                    'signal': signal,
                    'market_id': data.get('market_id'),
                    'bias': data.get('bias'),
                    'reason': data.get('reason'),
                    'message': data.get('message'),  # Error message if present
                    'timestamp': data.get('timestamp'),
                    'pair': data.get('pair'),
                    'price_at_creation': data.get('price_at_creation'),  # Price when market was created
                }
        except asyncio.TimeoutError:
            self.logger().error(
                f"Timeout fetching signal from {self.api_url}",
                exc_info=False
            )
            return None
        except aiohttp.ClientError as e:
            self.logger().error(
                f"Network error fetching signal: {e}",
                exc_info=False
            )
            return None
        except Exception as e:
            self.logger().error(
                f"Error fetching signal from API: {e}",
                exc_info=True
            )
            return None
    
    async def _poll_api_loop(self):
        """
        Background task that polls the API at regular intervals.
        """
        while True:
            try:
                signal_data = await self._fetch_signal()
                if signal_data is not None:
                    self._last_api_response = signal_data
                    self._current_signal = signal_data['signal']
                # If fetch failed, keep previous signal_data
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger().error(
                    f"Error in API polling loop: {e}",
                    exc_info=True
                )
            
            await asyncio.sleep(self.poll_interval)
    
    def get_signal(self) -> int:
        """
        Override base method to return signal from API.
        
        Returns:
            1 for long, -1 for short, 0 for no-trade
        """
        return self._current_signal
    
    def check_and_set_leverage(self):
        """
        Override to set leverage and position mode.
        """
        if not self.set_leverage_flag:
            for connector in self.connectors.values():
                for trading_pair in connector.trading_pairs:
                    connector.set_position_mode(self.position_mode)
                    connector.set_leverage(trading_pair=trading_pair, leverage=self.leverage)
            self.set_leverage_flag = True
    
    def close_open_positions(self):
        """
        Override to close all open positions when stopping.
        """
        if not self.is_perpetual:
            return
        
        for connector_name, connector in self.connectors.items():
            for trading_pair, position in connector.account_positions.items():
                from hummingbot.core.data_type.common import PositionSide, PositionAction
                if position.position_side == PositionSide.LONG:
                    self.sell(connector_name=connector_name,
                              trading_pair=position.trading_pair,
                              amount=abs(position.amount),
                              order_type=OrderType.MARKET,
                              price=connector.get_mid_price(position.trading_pair),
                              position_action=PositionAction.CLOSE)
                elif position.position_side == PositionSide.SHORT:
                    self.buy(connector_name=connector_name,
                             trading_pair=position.trading_pair,
                             amount=abs(position.amount),
                             order_type=OrderType.MARKET,
                             price=connector.get_mid_price(position.trading_pair),
                             position_action=PositionAction.CLOSE)
    
    def handle_profitable_position(self, executor: PositionExecutor, reason: str) -> bool:
        """
        Handle a profitable position by activating trailing stop and removing time limit.
        For unprofitable positions, close them.
        
        Args:
            executor: The position executor to handle
            reason: Reason for handling (e.g., "market_id change", "signal flip")
        
        Returns:
            True if position was handled (profitable and kept open), False if closed
        """
        try:
            # Skip if executor is already closed or shutting down
            if executor.is_closed or executor.status == RunnableStatus.SHUTTING_DOWN:
                return False
            
            # Check profitability
            pnl_pct = executor.trade_pnl_pct
            is_profitable = pnl_pct > Decimal("0")
            
            if is_profitable:
                # Profitable position: manually activate trailing stop and remove time limit
                if executor.config.triple_barrier_config.trailing_stop:
                    trailing_delta = executor.config.triple_barrier_config.trailing_stop.trailing_delta
                    # Only manually activate once if not already activated (bypass normal activation threshold)
                    if executor._trailing_stop_trigger_pct is None:
                        executor._trailing_stop_trigger_pct = pnl_pct - trailing_delta
                        # Remove time limit for profitable positions
                        executor.config.triple_barrier_config.time_limit = None
                        self.logger().info(
                            f"Executor {executor.config.id} is profitable ({pnl_pct:.4%}). "
                            f"Manually activated trailing stop and removed time limit due to {reason} (trigger: {executor._trailing_stop_trigger_pct:.4%})"
                        )
                else:
                    # Remove time limit even if trailing stop not configured
                    executor.config.triple_barrier_config.time_limit = None
                    self.logger().info(
                        f"Executor {executor.config.id} is profitable ({pnl_pct:.4%}). "
                        f"Removed time limit due to {reason}, keeping position open."
                    )
                return True
            else:
                # Not profitable: check if loss is small enough to keep open
                small_loss_threshold = Decimal(str(self.small_loss_threshold))
                if pnl_pct > small_loss_threshold:
                    # Small loss: keep position open to allow recovery
                    self.logger().info(
                        f"Executor {executor.config.id} has small loss ({pnl_pct:.4%} > {small_loss_threshold:.4%}). "
                        f"Keeping position open due to {reason} to allow recovery."
                    )
                    return True
                
                # Larger loss: close the position
                amount_to_close = executor.amount_to_close
                if amount_to_close > Decimal("0"):
                    executor.place_close_order_and_cancel_open_orders(
                        close_type=CloseType.EARLY_STOP
                    )
                    self.logger().info(
                        f"Closing executor {executor.config.id} due to {reason} (PNL: {pnl_pct:.4%}, amount_to_close: {amount_to_close})"
                    )
                elif executor.open_filled_amount > Decimal("0"):
                    executor.early_stop(keep_position=False)
                    self.logger().info(
                        f"Stopped executor {executor.config.id} (position already closed) due to {reason}"
                    )
                else:
                    executor.early_stop(keep_position=False)
                    self.logger().info(
                        f"Stopped executor {executor.config.id} (no open position) due to {reason}"
                    )
                return False
        except Exception as e:
            self.logger().error(f"Error handling executor {executor.config.id}: {e}", exc_info=True)
            return False
    
    def check_rsi_and_activate_trailing_stop(self, executor: PositionExecutor) -> bool:
        """
        Check RSI for an active position and activate trailing stop if RSI reaches overbought/oversold.
        
        Args:
            executor: The position executor to check
            
        Returns:
            True if trailing stop was activated, False otherwise
        """
        try:
            # Skip if executor is already closed or shutting down
            if executor.is_closed or executor.status == RunnableStatus.SHUTTING_DOWN:
                return False
            
            # Skip if trailing stop already activated
            if executor._trailing_stop_trigger_pct is not None:
                return False
            
            # Check if candles are ready
            if not self.candles or not self.candles[0].ready:
                return False
            
            # Calculate RSI
            try:
                import pandas_ta as ta
                candles_df = self.candles[0].candles_df.copy()
                candles_df.ta.rsi(length=self.rsi_length, append=True)
                rsi_column = f"RSI_{self.rsi_length}"
                
                if rsi_column not in candles_df.columns:
                    return False
                
                last_candle = candles_df.iloc[-1]
                rsi = last_candle[rsi_column]
                
                # Check RSI conditions based on position side
                should_activate = False
                reason = ""
                
                if executor.config.side == TradeType.BUY:  # LONG position
                    if rsi > self.rsi_overbought:
                        should_activate = True
                        reason = f"RSI {rsi:.2f} > {self.rsi_overbought} (overbought)"
                elif executor.config.side == TradeType.SELL:  # SHORT position
                    if rsi < self.rsi_oversold:
                        should_activate = True
                        reason = f"RSI {rsi:.2f} < {self.rsi_oversold} (oversold)"
                
                if should_activate:
                    # Activate trailing stop
                    if executor.config.triple_barrier_config.trailing_stop:
                        trailing_delta = executor.config.triple_barrier_config.trailing_stop.trailing_delta
                        pnl_pct = executor.trade_pnl_pct
                        executor._trailing_stop_trigger_pct = pnl_pct - trailing_delta
                        # Remove time limit when RSI-based trailing stop activates
                        executor.config.triple_barrier_config.time_limit = None
                        self.logger().info(
                            f"Executor {executor.config.id} ({executor.config.side.name}): "
                            f"Activated trailing stop due to {reason} "
                            f"(PNL: {pnl_pct:.4%}, trigger: {executor._trailing_stop_trigger_pct:.4%})"
                        )
                        return True
                
                return False
            except Exception as e:
                self.logger().debug(f"Error calculating RSI for executor {executor.config.id}: {e}")
                return False
        except Exception as e:
            self.logger().error(f"Error checking RSI for executor {executor.config.id}: {e}", exc_info=True)
            return False
    
    def on_tick(self):
        """
        Override to add market_id tracking check before position creation.
        """
        try:
            self.clean_and_store_executors()
            # Clean up executor market_id mappings for closed executors
            active_executor_ids = {executor.config.id for executor in self.active_executors}
            closed_executor_ids = set(self._executor_market_ids.keys()) - active_executor_ids
            for executor_id in closed_executor_ids:
                self._executor_market_ids.pop(executor_id, None)
            if self.is_perpetual:
                self.check_and_set_leverage()
            
            # Get signal data first (needed for market_id and no-trade checks)
            if self._last_api_response is None:
                return
            
            signal = self._last_api_response.get('signal', 0)
            market_id = self._last_api_response.get('market_id')
            
            # Check for active positions
            active_executors = self.get_active_executors()
            
            # Check RSI for active positions and activate trailing stop if conditions are met
            if active_executors and self.all_candles_ready:
                for executor in active_executors:
                    self.check_rsi_and_activate_trailing_stop(executor)
            
            # Clear tracked market_id if all executors are closed
            # This ensures we can create new positions when all executors are closed
            if not active_executors:
                self._tracked_market_id = None
            
            # On every tick: check if any active market_signal executors have a different market_id
            # Keep profitable ones, close unprofitable ones
            # Only affect market_signal executors, not rsi_override executors
            if market_id and active_executors:
                # Find market_signal executors with a different market_id than current
                stale_market_executors = [
                    executor for executor in active_executors
                    if self._executor_market_ids.get(executor.config.id) != market_id
                ]
                for executor in stale_market_executors:
                    exec_market_id = self._executor_market_ids.get(executor.config.id, "N/A")
                    self.handle_profitable_position(executor, f"stale market_id ({exec_market_id} vs current {market_id})")
                
                # Update tracked market_id
                self._tracked_market_id = market_id
                    
            # If signal is no-trade (0) and we have open positions, handle based on profitability
            # This check also needs to happen before max_executors check
            # Only affect market_signal executors, not rsi_override executors
            if signal == 0 and active_executors:
                market_signal_executors = [
                    executor for executor in active_executors
                    if not self._executor_market_ids.get(executor.config.id, "").startswith("rsi_")
                ]
                if market_signal_executors:
                    self.logger().info(
                        f"Signal changed to no-trade. Checking {len(market_signal_executors)} market_signal position(s)."
                    )
                    for executor in market_signal_executors:
                        self.handle_profitable_position(executor, "signal flip")
                # Don't clear tracked market_id or return - allow new positions if RSI extremes allow it
            
            # Note: signal == 0 (no-trade) is handled in get_position_config() which checks RSI extremes
            # If RSI conditions aren't met, get_position_config() will return None
            
            # Check conditions before creating position (only after closing checks)
            if not (self.max_active_executors_condition and self.all_candles_ready and self.time_between_signals_condition):
                return
            
            # Market ID tracking: only apply to market_signal executors
            # RSI override executors don't need market_id tracking and can re-enter freely
            position_config = self.get_position_config()
            if not position_config:
                return
            
            executor_market_id = self._pending_market_id
            is_rsi_override = executor_market_id and executor_market_id.startswith("rsi_")
            
            # Close unprofitable RSI executors when valid opposite-direction market signal arrives
            # Only for market signals (signal 1 or -1), not no-trade (0)
            if not is_rsi_override and signal != 0 and active_executors:
                new_side = position_config.side
                unprofitable_opposite_rsi = [
                    executor for executor in active_executors
                    if (self._executor_market_ids.get(executor.config.id, "").startswith("rsi_")
                        and executor.config.side != new_side
                        and executor.trade_pnl_pct <= Decimal("0"))
                ]
                for executor in unprofitable_opposite_rsi:
                    self.logger().info(
                        f"Closing unprofitable RSI {executor.config.side.name} executor {executor.config.id} "
                        f"(PNL: {executor.trade_pnl_pct:.4%}) - valid {new_side.name} market signal received"
                    )
                    self.handle_profitable_position(executor, f"opposite market signal ({new_side.name})")
            
            # Prevent multiple RSI override executors - only allow one at a time
            if is_rsi_override:
                existing_rsi_executors = [
                    executor for executor in active_executors
                    if self._executor_market_ids.get(executor.config.id, "").startswith("rsi_")
                ]
                if existing_rsi_executors:
                    self.logger().debug(
                        f"Skipping RSI override - already have {len(existing_rsi_executors)} active RSI executor(s)"
                    )
                    return
            
            # Only apply market_id tracking to market_signal executors (not rsi_ prefixed)
            if not is_rsi_override and executor_market_id:
                # Check if we already have an active executor for this market_id
                existing_executor_for_market = None
                for executor in active_executors:
                    exec_market_id = self._executor_market_ids.get(executor.config.id)
                    # Only check non-rsi executors for duplicates
                    if exec_market_id == executor_market_id and not exec_market_id.startswith("rsi_"):
                        existing_executor_for_market = executor
                        break
                
                if existing_executor_for_market:
                    # Same market_id with active executor - skip to prevent duplicate
                    self.logger().debug(
                        f"Skipping market_id {executor_market_id} - already have active executor {existing_executor_for_market.config.id}"
                    )
                    return
                
                # Check if this market_id was already traded (prevents re-entry on signal flips)
                # Exception: Allow RSI override if reason is "No active bitcoin markets found"
                api_reason = self._last_api_response.get('reason', '') or ''
                allow_rsi_override_for_traded = (
                    signal == 0 and 
                    self.enable_rsi_override and 
                    api_reason and
                    "No active" in api_reason and "markets found" in api_reason
                )
                
                if executor_market_id in self._traded_market_ids and not allow_rsi_override_for_traded:
                    self.logger().debug(
                        f"Skipping market_id {executor_market_id} - already traded (signal may be flipping)"
                    )
                    return
                elif executor_market_id in self._traded_market_ids and allow_rsi_override_for_traded:
                    self.logger().info(
                        f"Allowing RSI override for already-traded market_id {executor_market_id} "
                        f"(reason: {api_reason})"
                    )
            
            # Check for profitable positions: only allow new executor in same direction
            # Apply to ALL executors (both market_signal and rsi_override) to prevent conflicting directions
            if active_executors and position_config:
                new_side = position_config.side
                profitable_executors = [
                    executor for executor in active_executors
                    if executor.trade_pnl_pct > Decimal("0")
                ]
                if profitable_executors:
                    # Check if any profitable executor is in opposite direction
                    for executor in profitable_executors:
                        if executor.config.side != new_side:
                            executor_type = "RSI override" if self._executor_market_ids.get(executor.config.id, "").startswith("rsi_") else "market_signal"
                            self.logger().info(
                                f"Skipping {new_side.name} position: existing profitable {executor.config.side.name} {executor_type} position "
                                f"(executor {executor.config.id}, PNL: {executor.trade_pnl_pct:.4%})"
                            )
                            return
            
            # Create executor
            signal_executor = PositionExecutor(
                strategy=self,
                config=position_config,
            )
            signal_executor.start()
            self.active_executors.append(signal_executor)
            
            # Store executor market_id (may be rsi_ prefixed)
            if executor_market_id:
                self._executor_market_ids[signal_executor.config.id] = executor_market_id
                
                # Update tracked market_id and mark as traded (only for regular market_signal executors)
                if not is_rsi_override:
                    self._tracked_market_id = executor_market_id
                    self._traded_market_ids.add(executor_market_id)
                    self.logger().info(
                        f"Created market_signal executor {signal_executor.config.id} for market_id {executor_market_id} "
                        f"(total active: {len(self.active_executors)}, total traded: {len(self._traded_market_ids)})"
                    )
                else:
                    self.logger().info(
                        f"Created RSI override executor {signal_executor.config.id} for market_id {executor_market_id} "
                        f"(total active: {len(self.active_executors)})"
                    )
            else:
                self.logger().info(
                    f"Created executor {signal_executor.config.id} "
                    f"(total active: {len(self.active_executors)})"
                )
            
            # Clear pending market_id after position creation
            self._pending_market_id = None
        except Exception as e:
            self.logger().error(f"Error in on_tick: {e}", exc_info=True)
    
    def get_position_config(self) -> Optional[PositionExecutorConfig]:
        """
        Override to create position config from API signal.
        """
        # Reset pending market_id at start to prevent stale values
        self._pending_market_id = None
        signal = self.get_signal()
        
        # Check if connector exists and is ready
        if self.exchange not in self.connectors:
            self.logger().error(f"Exchange {self.exchange} not found in connectors")
            return None
        
        connector = self.connectors[self.exchange]
        if not connector.ready:
            self.logger().debug(f"Connector {self.exchange} not ready yet")
            return None
        
        try:
            price = connector.get_mid_price(self.trading_pair)
            if price is None or price <= 0:
                self.logger().warning(f"Invalid price for {self.trading_pair}: {price}")
                return None
        except Exception as e:
            self.logger().error(f"Error getting mid price: {e}", exc_info=True)
            return None
        
        # Technical indicator confirmation: RSI, Bollinger Bands, and Volatility
        if self.candles and len(self.candles) > 0 and self.candles[0].ready:
            try:
                import pandas_ta as ta  # noqa: F401
                candles_df = self.candles[0].candles_df.copy()
                
                # Calculate RSI
                candles_df.ta.rsi(length=self.rsi_length, append=True)
                rsi_column = f"RSI_{self.rsi_length}"
                
                # Calculate Stochastic RSI for momentum/direction detection
                candles_df.ta.stochrsi(length=self.stoch_rsi_length, rsi_length=self.rsi_length, k=self.stoch_rsi_k, d=self.stoch_rsi_d, append=True)
                stoch_k_column = f"STOCHRSIk_{self.stoch_rsi_length}_{self.rsi_length}_{self.stoch_rsi_k}_{self.stoch_rsi_d}"
                stoch_d_column = f"STOCHRSId_{self.stoch_rsi_length}_{self.rsi_length}_{self.stoch_rsi_k}_{self.stoch_rsi_d}"
                
                # Calculate Bollinger Bands (using lower_std and upper_std like bollinger_v1.py)
                candles_df.ta.bbands(length=self.bb_length, lower_std=self.bb_std, upper_std=self.bb_std, append=True)
                
                # pandas_ta creates BBP column with format: BBP_{length}_{lower_std}_{upper_std}
                bbp_column = f"BBP_{self.bb_length}_{self.bb_std}_{self.bb_std}"
                bbl_column = f"BBL_{self.bb_length}_{self.bb_std}"
                bbu_column = f"BBU_{self.bb_length}_{self.bb_std}"
                
                # Use auto-generated BBP if available, otherwise calculate manually
                if bbp_column in candles_df.columns:
                    candles_df["BBP"] = candles_df[bbp_column]
                elif bbl_column in candles_df.columns and bbu_column in candles_df.columns:
                    # Manually calculate BBP (Bollinger Bands Percent): (close - lower_band) / (upper_band - lower_band)
                    band_range = candles_df[bbu_column] - candles_df[bbl_column]
                    # Avoid division by zero
                    band_range = band_range.replace(0, float('nan'))
                    candles_df["BBP"] = (candles_df["close"] - candles_df[bbl_column]) / band_range
                else:
                    # Log available BB columns for debugging
                    bb_columns = [c for c in candles_df.columns if 'BB' in c.upper()]
                    self.logger().warning(
                        f"Bollinger Bands columns not found. Expected BBL/BBU or BBP. "
                        f"Available BB columns: {bb_columns}"
                    )
                    candles_df["BBP"] = float('nan')
                
                # Calculate NATR (Normalized ATR) for volatility filtering
                if self.enable_volatility_filter:
                    natr = ta.natr(
                        candles_df["high"],
                        candles_df["low"],
                        candles_df["close"],
                        length=self.atr_length
                    )
                    candles_df["NATR"] = natr
                
                # Check if both RSI and BBP are available and valid
                rsi_available = rsi_column in candles_df.columns
                bbp_available = "BBP" in candles_df.columns
                bbp_valid = bbp_available and candles_df["BBP"].notna().any() if bbp_available else False
                
                if rsi_available and bbp_valid:
                    last_candle = candles_df.iloc[-1]
                    rsi = last_candle[rsi_column]
                    bbp = last_candle["BBP"]
                    
                    # Get Stochastic RSI for momentum/direction detection
                    # %K > %D = bullish momentum (RSI rising), %K < %D = bearish momentum (RSI falling)
                    stoch_k = last_candle.get(stoch_k_column, 50)
                    stoch_d = last_candle.get(stoch_d_column, 50)
                    rsi_rising = stoch_k > stoch_d
                    rsi_falling = stoch_k < stoch_d
                    stoch_info = f"StochRSI K={stoch_k:.1f} D={stoch_d:.1f}"
                    
                    # Volatility filter: Skip if volatility is too low (choppy market)
                    if self.enable_volatility_filter and "NATR" in candles_df.columns:
                        natr_value = last_candle["NATR"]
                        if natr_value < self.min_volatility_pct:
                            self.logger().info(
                                f"Skipping signal: NATR {natr_value:.3f}% < {self.min_volatility_pct}% (insufficient volatility, choppy market) RSI {rsi:.2f}, BBP {bbp:.3f}"
                            )
                            return None

                    # Handle no-trade signal: use RSI override if enabled
                    if signal == 0:
                        if self.enable_rsi_override:
                            # RSI extremely oversold + BBP at lower band + RSI rising (bouncing) → LONG
                            if rsi < self.rsi_extreme_oversold and bbp < self.bb_lower_threshold and rsi_rising:
                                signal = 1
                                # Use rsi_ prefixed market_id if available, otherwise generate one
                                base_market_id = self._last_api_response.get('market_id') if self._last_api_response else None
                                if base_market_id:
                                    self._pending_market_id = f"rsi_{base_market_id}"
                                else:
                                    # Generate a unique ID if no market_id available
                                    self._pending_market_id = f"rsi_{int(time.time())}"
                                self.logger().info(
                                    f"RSI override (no-trade): RSI {rsi:.2f} < {self.rsi_extreme_oversold} (extremely oversold), "
                                    f"{stoch_info} (rising), BBP {bbp:.3f} < {self.bb_lower_threshold}. Opening LONG."
                                )
                            # RSI extremely overbought + BBP at upper band + RSI falling (reversing) → SHORT
                            elif rsi > self.rsi_extreme_overbought and bbp > self.bb_upper_threshold and rsi_falling:
                                signal = -1
                                # Use rsi_ prefixed market_id if available, otherwise generate one
                                base_market_id = self._last_api_response.get('market_id') if self._last_api_response else None
                                if base_market_id:
                                    self._pending_market_id = f"rsi_{base_market_id}"
                                else:
                                    # Generate a unique ID if no market_id available
                                    self._pending_market_id = f"rsi_{int(time.time())}"
                                self.logger().info(
                                    f"RSI override (no-trade): RSI {rsi:.2f} > {self.rsi_extreme_overbought} (extremely overbought), "
                                    f"{stoch_info} (falling), BBP {bbp:.3f} > {self.bb_upper_threshold}. Opening SHORT."
                                )
                            else:
                                # No-trade signal and RSI/BBP don't meet criteria (including direction)
                                direction_info = "rising" if rsi_rising else "falling" if rsi_falling else "flat"
                                self.logger().info(
                                    f"Skipping no-trade signal: RSI {rsi:.2f}, {stoch_info} ({direction_info}), BBP {bbp:.3f} - conditions not met"
                                )
                                return None
                        else:
                            # No-trade signal and RSI override disabled
                            self.logger().debug(
                                f"No-trade signal: RSI override disabled, skipping position"
                            )
                            return None
                    
                    # Normal RSI filtering for API signals (1 or -1) - no override, just skip if unfavorable
                    # Use regular market_id for market signals
                    if self._pending_market_id is None:
                        self._pending_market_id = self._last_api_response.get('market_id') if self._last_api_response else None
                    
                    if signal == 1:
                        if rsi > self.rsi_overbought:
                            # Overbought: skip LONG
                            self.logger().info(
                                f"Skipping LONG signal: RSI {rsi:.2f} > {self.rsi_overbought} (overbought), {stoch_info}"
                            )
                            return None
                        if rsi_falling and rsi > self.rsi_oversold:
                            # RSI momentum bearish: wait for stabilization before LONG
                            self.logger().info(
                                f"Skipping LONG signal: RSI {rsi:.2f}, {stoch_info} (falling) - waiting for stabilization"
                            )
                            return None
                    elif signal == -1:
                        if rsi < self.rsi_oversold:
                            # Oversold: skip SHORT
                            self.logger().info(
                                f"Skipping SHORT signal: RSI {rsi:.2f} < {self.rsi_oversold} (oversold), {stoch_info}"
                            )
                            return None
                        if rsi_rising and rsi < self.rsi_overbought:
                            # RSI momentum bullish: wait for stabilization before SHORT
                            self.logger().info(
                                f"Skipping SHORT signal: RSI {rsi:.2f}, {stoch_info} (rising) - waiting for stabilization"
                            )
                            return None
                    
                    # Check BBP for the final signal (original or flipped)
                    if signal == 1:
                        if bbp > self.bb_upper_threshold:
                            self.logger().info(
                                f"Skipping LONG signal: BBP {bbp:.3f} > {self.bb_upper_threshold} (at upper Bollinger Band)"
                            )
                            return None
                    elif signal == -1:
                        if bbp < self.bb_lower_threshold:
                            self.logger().info(
                                f"Skipping SHORT signal: BBP {bbp:.3f} < {self.bb_lower_threshold} (at lower Bollinger Band)"
                            )
                            return None
                    
                    volatility_info = ""
                    if self.enable_volatility_filter and "NATR" in candles_df.columns:
                        natr_value = last_candle["NATR"]
                        volatility_info = f", NATR={natr_value:.3f}%"
                    
                    self.logger().debug(
                        f"Signal confirmed by indicators: RSI={rsi:.2f}, BBP={bbp:.3f}{volatility_info}"
                    )
                    self.logger().info(f"Tech checks passed: RSI={rsi:.2f}, BBP={bbp:.3f}{volatility_info}")
                else:
                    # Indicators calculated but columns missing - skip position
                    bb_columns = [c for c in candles_df.columns if 'BB' in c.upper()]
                    self.logger().warning(
                        f"Technical indicator columns missing (RSI: {rsi_available}, "
                        f"BBP available: {bbp_available}, BBP valid: {bbp_valid}). "
                        f"Available BB columns: {bb_columns}. Skipping position creation."
                    )
                    return None
            except Exception as e:
                # If candles are ready but indicators fail, don't create position
                self.logger().error(
                    f"Error calculating technical indicators: {e}. Skipping position creation for safety.",
                    exc_info=True
                )
                return None
        
        side = TradeType.BUY if signal == 1 else TradeType.SELL
        
        if self.open_order_type.is_limit_type():
            price = price * (1 - signal * self.open_order_slippage_buffer)
        
        try:
            # Get current timestamp, fallback to time.time() if not available
            try:
                timestamp = self.current_timestamp
            except (AttributeError, TypeError):
                timestamp = time.time()
            
            # Calculate position value from margin: position_value = margin * leverage
            position_value_usd = self.margin_usd * Decimal(self.leverage)
            # Calculate base amount: amount = position_value / price
            amount = position_value_usd / price
            
            position_config = PositionExecutorConfig(
                timestamp=timestamp,
                trading_pair=self.trading_pair,
                connector_name=self.exchange,
                side=side,
                amount=amount,
                entry_price=price,
                triple_barrier_config=self.triple_barrier_conf,
                leverage=self.leverage,
            )
            return position_config
        except Exception as e:
            self.logger().error(f"Error creating position config: {e}", exc_info=True)
            return None
    
    def format_status(self) -> str:
        """
        Override to display API signal status, bias, reason, and active positions.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        # API Status Section
        lines.extend(["\n################################## API Signal Status ##################################"])
        if self._last_api_response:
            response = self._last_api_response
            signal = response.get('signal', 0)
            signal_text = "LONG" if signal == 1 else "SHORT" if signal == -1 else "NO-TRADE"
            lines.extend([f"Signal: {signal_text} ({signal})"])
            lines.extend([f"Pair: {response.get('pair', 'N/A')}"])
            lines.extend([f"Market ID: {response.get('market_id', 'N/A')}"])
            lines.extend([f"Bias: {response.get('bias', 'N/A')}%"])
            lines.extend([f"Timestamp: {response.get('timestamp', 'N/A')}"])
            if response.get('reason'):
                lines.extend([f"Reason: {response.get('reason')}"])
            if response.get('message'):
                lines.extend([f"Message: {response.get('message')}"])
        else:
            lines.extend(["No API response received yet."])
        
        lines.extend([f"Tracked Market ID: {self._tracked_market_id or 'None'}"])
        lines.extend([f"Traded Market IDs: {len(self._traded_market_ids)} ({', '.join(list(self._traded_market_ids)[:5])}{'...' if len(self._traded_market_ids) > 5 else ''})"])
        lines.extend([f"Poll Interval: {self.poll_interval}s"])
        
        # Executor types summary (market_signal vs rsi_override)
        market_signal_count = 0
        rsi_override_count = 0
        for executor in self.active_executors:
            exec_market_id = self._executor_market_ids.get(executor.config.id, "")
            if exec_market_id.startswith("rsi_"):
                rsi_override_count += 1
            else:
                market_signal_count += 1
        if market_signal_count > 0 or rsi_override_count > 0:
            lines.extend([f"Active Executors: market_signal={market_signal_count}, rsi_override={rsi_override_count}"])
        
        # Closed Executors
        if len(self.stored_executors) > 0:
            lines.extend(["\n################################## Closed Executors ##################################"])
            for executor in self.stored_executors:
                market_id = self._executor_market_ids.get(executor.config.id, "N/A")
                lines.extend([f"|Timestamp: {executor.config.timestamp} | Market id: {market_id}"])
                lines.extend(executor.to_format_status())
                lines.extend([
                    "-----------------------------------------------------------------------------------------------------------"])
        
        # Active Executors
        if len(self.active_executors) > 0:
            lines.extend(["\n################################## Active Executors ##################################"])
            for executor in self.active_executors:
                market_id = self._executor_market_ids.get(executor.config.id, "N/A")
                lines.extend([f"|Timestamp: {executor.config.timestamp} | Market id: {market_id}"])
                lines.extend(executor.to_format_status())
        else:
            lines.extend(["\n################################## Active Executors ##################################"])
            lines.extend(["No active executors."])
        
        return "\n".join(lines)
    
    async def on_stop(self):
        """
        Override to stop API polling task and close HTTP client.
        """
        # Cancel API polling task
        if self._api_polling_task and not self._api_polling_task.done():
            self._api_polling_task.cancel()
        
        # Close HTTP client - store reference to avoid race conditions
        http_client = self._http_client
        if http_client is not None and not http_client.closed:
            try:
                await http_client.close()
            except Exception as e:
                self.logger().error(f"Error closing HTTP client: {e}", exc_info=True)
        
        # Call parent on_stop (synchronous method, no await needed)
        super().on_stop()

