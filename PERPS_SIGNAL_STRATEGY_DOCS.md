# Perps Signal Strategy Documentation

## Overview

The **Perps Signal Strategy** is an automated directional trading strategy for perpetual futures contracts that executes trades based on signals from an external API. The strategy polls a signal API endpoint at regular intervals and automatically opens, manages, and closes positions based on the received signals.

### Key Features

- **API-Driven Trading**: Automatically fetches trading signals from an external API endpoint
- **Market ID Tracking**: Prevents duplicate positions for the same market opportunity
- **Intelligent Position Management**: Handles profitable positions differently from unprofitable ones when signals change
- **Multiple Active Positions**: Supports multiple concurrent positions for different market IDs
- **Technical Indicator Confirmation**: Uses RSI and Bollinger Bands to filter out unfavorable entry conditions
- **RSI Override**: Optionally flips signal direction when RSI strongly contradicts the API signal
- **No-Trade Signal Trading**: Uses RSI extremes (30/70) with BBP filter to trade when API signal is no-trade
- **Triple Barrier Risk Management**: Implements stop-loss, take-profit, time-limit, and trailing stop protection
- **Automatic Position Protection**: Activates trailing stops for profitable positions when signals flip

## Architecture

The strategy extends `DirectionalStrategyBase` and uses `PositionExecutor` instances to manage individual trading positions. Each position is protected by the Triple Barrier Method, which includes:

- **Stop Loss**: Closes position if loss exceeds threshold
- **Take Profit**: Closes position if profit exceeds threshold
- **Time Limit**: Closes position after a specified duration
- **Trailing Stop**: Dynamically adjusts stop-loss to lock in profits

## Configuration

### Configuration File

The strategy is configured via a YAML file (e.g., `conf/scripts/conf_perps_signal_strategy_1.yml`):

```yaml
script_file_name: perps_signal_strategy.py
exchange: hyperliquid_perpetual
trading_pair: BTC-USD
api_url: http://agents.eternax.ai/perps_signal
poll_interval: 30.0
max_executors: 2
position_mode: ONEWAY
leverage: 30
margin_usd: '100'
stop_loss: 0.02
take_profit: 0.02
time_limit: 3600
open_order_type: 1
take_profit_order_type: 1
stop_loss_order_type: 1
time_limit_order_type: 1
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exchange` | string | `hyperliquid_perpetual` | Exchange connector name |
| `trading_pair` | string | `BTC-USD` | Trading pair symbol |
| `api_url` | string | `http://agents.eternax.ai/perps_signal` | Signal API endpoint URL |
| `poll_interval` | float | `30.0` | API polling interval in seconds (must be > 0) |
| `max_executors` | int | `1` | Maximum number of concurrent active positions |
| `position_mode` | string | `ONEWAY` | Position mode: `ONEWAY` or `HEDGE` |
| `leverage` | int | `10` | Leverage multiplier (must be > 0) |
| `margin_usd` | Decimal | `10` | Margin amount in USD. Position value = margin × leverage |
| `stop_loss` | float | `0.03` | Stop loss percentage (e.g., 0.03 = 3%) |
| `take_profit` | float | `0.01` | Take profit percentage (e.g., 0.01 = 1%) |
| `time_limit` | int | `3600` | Maximum position duration in seconds |
| `open_order_type` | int | `1` | Order type for opening positions (1=Market, 2=Limit) |
| `take_profit_order_type` | int | `1` | Order type for take profit (1=Market, 2=Limit) |
| `stop_loss_order_type` | int | `1` | Order type for stop loss (1=Market, 2=Limit) |
| `time_limit_order_type` | int | `1` | Order type for time limit exit (1=Market, 2=Limit) |
| `rsi_length` | int | `14` | RSI period length for technical indicator confirmation |
| `rsi_overbought` | float | `70.0` | RSI overbought threshold (skip LONG if RSI > this) |
| `rsi_oversold` | float | `30.0` | RSI oversold threshold (skip SHORT if RSI < this) |
| `bb_length` | int | `20` | Bollinger Bands period length |
| `bb_std` | float | `2.0` | Bollinger Bands standard deviation |
| `bb_upper_threshold` | float | `0.8` | BBP upper threshold (skip LONG if BBP > this) |
| `bb_lower_threshold` | float | `0.2` | BBP lower threshold (skip SHORT if BBP < this) |
| `enable_rsi_override` | bool | `false` | If RSI strongly contradicts signal, flip direction instead of skipping |
| `candles_interval` | string | `"5m"` | Candles interval for technical indicators |
| `candles_exchange` | string | `null` | Exchange for candles data (defaults to trading exchange) |
| `small_loss_threshold` | float | `-0.005` | Keep unprofitable positions open if PNL > this threshold (e.g., -0.005 = -0.5%) |
| `enable_volatility_filter` | bool | `true` | Enable volatility filter to avoid choppy markets |
| `atr_length` | int | `14` | ATR period length in candles for volatility calculation |
| `min_volatility_pct` | float | `0.15` | Minimum NATR percentage to enter position (e.g., 0.15 = 0.15%) |

### Position Value Calculation

The strategy calculates position size based on margin and leverage:

```
Position Value (USD) = margin_usd × leverage
Base Amount = Position Value / Entry Price
```

**Example**: With `margin_usd = 100` and `leverage = 30`:
- Position Value = 100 × 30 = $3,000
- If BTC price is $90,000, Base Amount = 3,000 / 90,000 = 0.0333 BTC

## How It Works

### 1. API Polling

The strategy continuously polls the signal API at the configured interval:

- **Polling Frequency**: Every `poll_interval` seconds (default: 30s)
- **Error Handling**: Network errors and timeouts are logged but don't stop the strategy
- **Signal Persistence**: If an API call fails, the strategy uses the last successful response

### 2. Signal Processing

The API returns signals in the following format:

```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "action": "long",
  "bias": 15.3,
  "reason": "Strong upward bias...",
  "market_id": "market_abc123"
}
```

**Note**: The API may include a `price_at_creation` field, but this is not used by the strategy for validation. The strategy uses technical indicators (RSI and Bollinger Bands) based on exchange candles for more reliable entry validation.

**Signal Mapping**:
- `action: "long"` → Signal = `1` (BUY)
- `action: "short"` → Signal = `-1` (SELL)
- `action: "no-trade"` → Signal = `0` (No position, but may trade based on RSI extremes)

### 3. Market ID Tracking

The strategy uses `market_id` to track unique trading opportunities:

- **Prevents Duplicates**: Won't open a new position for a `market_id` that already has an active executor
- **Prevents Re-entry**: Won't open a new position for a `market_id` that has already been traded (even if all positions have closed)
- **Tracks History**: Maintains a set of all `market_ids` that have been traded to prevent re-entry on signal flips

This prevents the strategy from re-entering markets that have already been traded, avoiding confusion from signal flips or repeated signals for the same market opportunity.

### 4. Position Creation Logic

A new position is created when:

1. ✅ API response is available
2. ✅ Signal is not `0` (no-trade) **OR** RSI extremes with BBP filter allow trading (RSI < 30 + BBP < lower_threshold for LONG, or RSI > 70 + BBP > upper_threshold for SHORT)
3. ✅ Max executors condition is met (`active_executors < max_executors`)
4. ✅ All candles are ready (for technical indicator confirmation)
5. ✅ Time between signals condition is met
6. ✅ No active executor exists for the current `market_id`
7. ✅ `market_id` has not been previously traded (prevents re-entry on signal flips)
8. ✅ Connector is ready
9. ✅ Valid price is available
10. ✅ **Technical indicator confirmation passes** (RSI and Bollinger Bands filters, or RSI override applies)
11. ✅ **Volatility filter passes** (if enabled, NATR >= min_volatility_pct)

**Technical Indicator Confirmation**:

The strategy uses RSI (Relative Strength Index) and Bollinger Band Percent (BBP) to filter out unfavorable entry conditions and optionally override signals:

#### Normal Filtering (Always Active)

- **LONG signals are filtered if**:
  - RSI > `rsi_overbought` (default: 70, commonly adjusted to 60) - Price is overbought
  - BBP > `bb_upper_threshold` (default: 0.8, commonly adjusted to 0.6) - Price is at upper Bollinger Band

- **SHORT signals are filtered if**:
  - RSI < `rsi_oversold` (default: 30, commonly adjusted to 40) - Price is oversold
  - BBP < `bb_lower_threshold` (default: 0.2, commonly adjusted to 0.4) - Price is at lower Bollinger Band

#### RSI Override (Optional, when `enable_rsi_override: true`)

When RSI strongly contradicts the signal, the strategy can flip the direction instead of skipping:

- **LONG signal + RSI > overbought + 10** (e.g., RSI > 70):
  - If override enabled: Flip to SHORT
  - If override disabled: Skip LONG

- **SHORT signal + RSI < oversold - 10** (e.g., RSI < 30):
  - If override enabled: Flip to LONG
  - If override disabled: Skip SHORT

**Example**: API signal is SHORT, but RSI is 37 (< 40 oversold threshold). Normal filter skips SHORT. If RSI drops to 28 (< 30 = oversold - 10) and override is enabled, strategy flips to LONG instead.

#### No-Trade Signal Trading (Always Active)

When the API signal is `0` (no-trade), the strategy can still open positions based on RSI extremes filtered by BBP:

- **RSI < 30 (strongly oversold) AND BBP < lower_threshold** → Opens LONG position
- **RSI > 70 (strongly overbought) AND BBP > upper_threshold** → Opens SHORT position
- Otherwise: No position (returns `None`)

This allows the strategy to capitalize on extreme RSI conditions even when the API provides no explicit signal.

**Note**: The `price_at_creation` field from the API is not used for validation because it comes from CoinGecko, which may have different prices than the actual trading exchange (e.g., Hyperliquid). Technical indicators based on exchange candles provide more reliable entry validation.

### 5. Position Management

The strategy handles three scenarios when signals change:

#### Scenario A: Market ID Changes

When a new `market_id` is detected while positions are open:

1. **Profitable Positions**:
   - Manually activates trailing stop (bypasses normal activation threshold)
   - Removes time limit (allows position to run indefinitely)
   - Position continues under trailing stop protection

2. **Unprofitable Positions**:
   - Immediately closes the position
   - Updates `_tracked_market_id` to the new market ID
   - Allows new position to be created for the new market ID (if not already traded)

#### Scenario B: Signal Flips to No-Trade

When signal changes to `0` (no-trade) while positions are open:

1. **Profitable Positions**:
   - Manually activates trailing stop
   - Removes time limit
   - Position continues under trailing stop protection

2. **Unprofitable Positions**:
   - Immediately closes the position
   - Strategy waits for signal to return before creating new positions

#### Scenario C: All Executors Close

When all positions are closed:

- `_tracked_market_id` is cleared to `None`
- Strategy is ready to create new positions for any `market_id`

### 6. Risk Management

Each position is protected by the **Triple Barrier Method**:

#### Stop Loss
- Closes position if unrealized loss exceeds configured percentage
- Example: With `stop_loss: 0.02` (2%), position closes if loss ≥ 2%

#### Take Profit
- Closes position if unrealized profit exceeds configured percentage
- Example: With `take_profit: 0.02` (2%), position closes if profit ≥ 2%

#### Time Limit
- Closes position after specified duration
- Example: With `time_limit: 3600`, position closes after 1 hour
- **Note**: Time limit is automatically removed for profitable positions when signals change

#### Trailing Stop
- **Default Activation**: Activates when profit exceeds 0.3% (default from `DirectionalStrategyBase`)
- **Manual Activation**: For profitable positions when signals change, trailing stop activates immediately regardless of profit amount
- **Trailing Behavior**: Locks in profits by adjusting stop-loss as price moves favorably
- **Default Trailing Delta**: 0.1% (default from `DirectionalStrategyBase`)

**Trailing Stop Example**:
- Position is profitable at 0.5%
- Trailing stop activates (manually or at 0.3% threshold)
- Trigger set at: 0.5% - 0.1% = 0.4%
- If price moves to 1.0%, trigger updates to: 1.0% - 0.1% = 0.9%
- If price drops below trigger (e.g., 0.85%), position closes

## Key Methods

### `handle_profitable_position(executor, reason)`

Handles positions when signals change or market IDs change:

- **Inputs**:
  - `executor`: The `PositionExecutor` instance to handle
  - `reason`: String describing why position is being handled (e.g., "market_id change", "signal flip")

- **Behavior**:
  - If profitable: Activates trailing stop and removes time limit
  - If unprofitable: Closes the position immediately

- **Returns**: `True` if position kept open, `False` if closed

### `on_tick()`

Main strategy loop executed every tick:

1. Cleans closed executors
2. Updates market ID mappings
3. Checks for market ID changes
4. Checks for signal flips to no-trade
5. Validates conditions for new position creation
6. Creates new positions when appropriate

### `get_position_config()`

Creates a `PositionExecutorConfig` from the current API signal:

- Handles signal `0` (no-trade) by checking RSI extremes with BBP filter
- Checks connector readiness
- Validates price
- Applies technical indicator filters (RSI and Bollinger Bands)
- Applies RSI override if enabled and RSI strongly contradicts signal
- Applies volatility filter if enabled
- Calculates position size from margin and leverage
- Returns config or `None` if validation fails

## Example Scenarios

### Scenario 1: First Position

1. Strategy starts with no active positions
2. API returns: `action: "long"`, `market_id: "market_001"`
3. Strategy creates LONG position
4. `_tracked_market_id = "market_001"`

### Scenario 2: Market ID Change with Profitable Position

1. Active LONG position is profitable (0.5%)
2. API returns: `action: "short"`, `market_id: "market_002"`
3. Strategy detects new `market_id`
4. Position is profitable → Trailing stop activated, time limit removed
5. Position continues running under trailing stop
6. Strategy creates new SHORT position for `market_002`
7. Now have 2 active positions (one LONG with trailing stop, one SHORT)

### Scenario 3: Signal Flips to No-Trade

1. Active position is profitable (0.2%)
2. API returns: `action: "no-trade"`, `market_id: "market_001"`
3. Strategy detects signal flip
4. Position is profitable → Trailing stop activated, time limit removed
5. Position continues running under trailing stop
6. No new positions created (signal is no-trade)

### Scenario 4: Unprofitable Position on Signal Change

1. Active position is unprofitable (-0.5%)
2. API returns: `action: "short"`, `market_id: "market_002"`
3. Strategy detects new `market_id`
4. Position is unprofitable → Position closed immediately
5. Strategy creates new SHORT position for `market_002`

### Scenario 5: RSI Override

1. API returns: `action: "short"`, `market_id: "market_003"`
2. RSI is 28 (< 30 = oversold - 10)
3. `enable_rsi_override: true` is configured
4. Strategy detects strong oversold condition contradicts SHORT signal
5. Signal flipped to LONG → Opens LONG position instead of SHORT
6. Log: "RSI override: SHORT signal contradicted by RSI 28.00 < 30.0 (strongly oversold). Flipping to LONG."

### Scenario 6: No-Trade Signal with RSI Extreme

1. API returns: `action: "no-trade"`, `market_id: "market_004"`
2. RSI is 25 (< 30) and BBP is 0.35 (< 0.4 lower_threshold)
3. Strategy detects strongly oversold condition with favorable BBP
4. Opens LONG position based on RSI/BBP conditions
5. Log: "No-trade signal: RSI 25.00 < 30 (strongly oversold) and BBP 0.350 < 0.4 (at lower band). Opening LONG."

## Status Display

The strategy provides detailed status information via `format_status()`:

```
################################## API Signal Status ##################################
Signal: LONG (1)
Pair: BTC-USDC
Market ID: market_abc123
Bias: 15.3%
Timestamp: 2024-01-15T10:30:00.000Z
Reason: Strong upward bias...
Tracked Market ID: market_abc123
Traded Market IDs: 3 (market_001, market_002, market_abc123)
Poll Interval: 30.0s

################################## Active Executors ##################################
|Signal id: 1705312200.0
| Trading Pair: BTC-USD | Exchange: hyperliquid_perpetual | Side: TradeType.BUY
Entry price: 42150.000000 | Current price: 42500.000000 | Amount: 0.0711 BTC
Unrealized PNL: 24.87 USD | PNL (%): 0.83% | Status: RUNNING
```

## Troubleshooting

### Positions Not Opening

**Check if**:
1. API is returning valid signals (check logs for API responses)
2. Signal is not `0` (no-trade) **OR** RSI extremes with BBP filter allow trading
3. `max_executors` limit not reached
4. Connector is ready
5. Candles are ready for technical indicator calculation
6. Technical indicator confirmation passes (RSI and BBP filters)
7. Volatility filter passes (if enabled, check NATR >= min_volatility_pct)
8. No active executor exists for current `market_id`
9. RSI override conditions are met (if expecting override behavior)

### Positions Closing Prematurely

**Possible Causes**:
1. Stop loss triggered (check `stop_loss` setting)
2. Take profit triggered (check `take_profit` setting)
3. Time limit reached (check `time_limit` setting)
4. Trailing stop triggered (check trailing stop configuration)
5. Signal changed and position was unprofitable

### Duplicate Positions for Same Market ID

**Should Not Happen**: The strategy prevents this by checking `_executor_market_ids` before creating positions. If this occurs, check:
1. Executor cleanup is working correctly
2. `_executor_market_ids` mapping is being maintained properly

### Trailing Stop Not Activating

**Normal Behavior**: Trailing stop activates at 0.3% profit by default, OR manually when signals change for profitable positions.

**To Verify**:
- Check logs for "Manually activated trailing stop" messages
- Check executor status for trailing stop state

## Best Practices

1. **Leverage**: Start with lower leverage (10-20x) and increase gradually
2. **Margin**: Use appropriate margin amounts based on your risk tolerance
3. **Stop Loss/Take Profit**: Set based on expected volatility (2-3% is common)
4. **Time Limit**: Set based on typical signal duration (1 hour is a good default)
5. **Max Executors**: Set based on capital and risk management (1-3 is typical)
6. **Poll Interval**: 30 seconds is a good balance between responsiveness and API load

## API Requirements

The strategy expects the API to return JSON with the following structure:

**Success Response**:
```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "action": "long",
  "bias": 15.3,
  "reason": "Strong upward bias...",
  "market_id": "market_abc123"
}
```

**Note**: The API may include a `price_at_creation` field, but this is not used by the strategy. The strategy uses technical indicators (RSI and Bollinger Bands) based on exchange candles for more reliable entry validation.

**Error Response** (still returns 200 status):
```json
{
  "error": "Failed to generate perps signal",
  "details": "Error message here"
}
```

The strategy handles both success and error responses gracefully.

## Technical Details

### Market ID Tracking State

The strategy maintains three tracking mechanisms:

1. **`_tracked_market_id`**: The current market ID being tracked (set when position is created)
2. **`_traded_market_ids`**: Set of all market IDs that have been traded (prevents re-entry)
3. **`_executor_market_ids`**: Dictionary mapping executor IDs to market IDs (for active executors)

### Position Executor Lifecycle

1. **Creation**: `PositionExecutor` created with config from `get_position_config()`
2. **Start**: Executor started and added to `active_executors`
3. **Monitoring**: Executor manages position (stop-loss, take-profit, time-limit, trailing-stop)
4. **Closure**: Executor closes position when barrier is hit or manually stopped
5. **Cleanup**: Executor moved to `stored_executors` and market ID mappings cleaned up

### Error Handling

- **API Errors**: Logged but don't stop strategy (uses last successful response)
- **Network Errors**: Logged and retried on next poll
- **Position Creation Errors**: Logged and position creation skipped
- **Executor Errors**: Logged and executor continues with error handling

## Configuration Refresh

To refresh configuration without rebuilding Docker container:

1. Edit configuration file: `conf/scripts/conf_perps_signal_strategy_1.yml`
2. In Hummingbot CLI: `stop` (stops the strategy)
3. In Hummingbot CLI: `start` (restarts with new configuration)

The configuration is loaded from the mounted volume, so changes persist across container restarts.

## Advanced Features

### RSI Override Mechanism

The RSI override feature (`enable_rsi_override: true`) allows the strategy to flip signal direction when RSI strongly contradicts the API signal:

**How it works**:
- **Normal filtering**: RSI thresholds filter out unfavorable entries (e.g., skip LONG if RSI > 60)
- **Strong override**: When RSI exceeds the threshold by 10 points (e.g., RSI > 70 for LONG signal), the strategy can flip the direction instead of skipping

**Use cases**:
- API signal may be delayed or incorrect
- Strong RSI extremes indicate better trading opportunities
- Allows strategy to capitalize on mean reversion opportunities

**Example**: API says SHORT, but RSI drops to 28 (strongly oversold). With override enabled, strategy opens LONG instead, betting on a bounce.

### No-Trade Signal Trading

When the API returns `no-trade`, the strategy can still trade based on RSI extremes:

**Conditions for LONG**:
- RSI < 30 (strongly oversold)
- BBP < lower_threshold (price at lower Bollinger Band)

**Conditions for SHORT**:
- RSI > 70 (strongly overbought)
- BBP > upper_threshold (price at upper Bollinger Band)

This feature allows the strategy to remain active even when the API provides no explicit signal, capitalizing on extreme technical conditions.

## Summary

The Perps Signal Strategy is a sophisticated automated trading system that:

- ✅ Fetches signals from an external API
- ✅ Manages multiple concurrent positions
- ✅ Prevents duplicate positions for the same market
- ✅ Protects profitable positions when signals change
- ✅ Closes unprofitable positions immediately (or keeps small losses open for recovery)
- ✅ Validates signals using technical indicators (RSI and BBP)
- ✅ Optionally overrides signals based on strong RSI contradictions
- ✅ Trades on RSI extremes even when API signal is no-trade
- ✅ Filters out low-volatility (choppy) markets
- ✅ Implements comprehensive risk management

The strategy is designed to be robust, handling errors gracefully while maintaining position safety through the Triple Barrier Method and intelligent position management.
