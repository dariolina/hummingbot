# Perps Signal API Documentation

## Overview

The Perps Signal API provides trading signals for perpetual futures contracts based on agent consensus from prediction markets. The endpoint uses the same calculations as the dashboard gauges to determine long/short positions.

**Base URL**: `http://agents.eternax.ai`

## Endpoint

### GET `/perps_signal`

Get a perpetual futures trading signal for Bitcoin based on active prediction markets.

#### Request

**Method**: `GET`

**URL**: `/perps_signal`

**Query Parameters**: None

**Headers**: None required

#### Response

**Status Code**: `200 OK` (always returns 200, even for errors)

**Content-Type**: `application/json`

#### Success Response

When a valid signal is generated, the response contains:

```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "action": "long",
  "bias": 15.3,
  "close_time": "2024-01-15T11:00:00.000Z",
  "reason": "Strong upward bias (15.3%) with sufficient movement probability (25.4%). Consensus confirmed.",
  "market_id": "market_abc123",
  "price_at_creation": 42150.50
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `pair` | string | Trading pair (currently always "BTC-USDC") |
| `timestamp` | string | ISO 8601 timestamp when the signal was generated |
| `action` | string | Trading action: `"long"`, `"short"`, or `"no-trade"` |
| `bias` | number | Directional bias percentage (-100 to +100). Positive = bullish, negative = bearish |
| `close_time` | string | ISO 8601 timestamp when the prediction market closes |
| `reason` | string | Human-readable explanation of the signal |
| `market_id` | string \| null | ID of the primary market used for the signal |
| `price_at_creation` | number \| null | Asset price when the market was created (entry reference price) |

**Action Values**:
- `"long"`: Open a long position (buy)
- `"short"`: Open a short position (sell)
- `"no-trade"`: No signal available (see error response format)

#### Error Response

When no signal is available, the response uses the same structure but with an `error` field:

```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "error": "No signal available",
  "message": "Bias (8.5%) below threshold (12%). Signal too weak.",
  "action": "no-trade",
  "bias": 8.5,
  "market_id": "market_abc123"
}
```

**Error Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `pair` | string | Trading pair |
| `timestamp` | string | ISO 8601 timestamp when the response was generated |
| `error` | string | Always `"No signal available"` |
| `message` | string | Detailed explanation of why no signal is available |
| `action` | string | Always `"no-trade"` |
| `bias` | number \| null | Current bias percentage, or `null` if markets are unavailable |
| `market_id` | string \| null | ID of the primary market, or `null` if unavailable |

#### Signal Generation Logic

The endpoint uses the following criteria to generate signals:

1. **Market Requirements**:
   - Both up and down markets must exist for the asset
   - Markets must be at least 5 minutes old
   - Markets must not have already closed (current time < close time)
   - At least one market must have ≥7 bets, and the other must have ≥3 bets
   - Market thresholds must be ≥0.15% away from creation price

2. **Bias Thresholds** (B_MIN):
   - **BTC**: 12% minimum bias required
   - **ETH**: 16% minimum bias required
   - **SOL**: 20% minimum bias required

3. **Movement Probability Thresholds** (M_MIN):
   - **BTC**: 18% minimum movement probability
   - **ETH**: 14% minimum movement probability
   - **SOL**: 12% minimum movement probability

4. **Consensus Requirements**:
   - For **LONG**: Bias must be positive AND greater than the down market's yes probability
   - For **SHORT**: Absolute bias must be greater than the up market's yes probability

5. **Bias Calculation**:
   ```
   bias = p_up - p_down
   ```
   Where:
   - `p_up` = probability of upward movement (from up market)
   - `p_down` = probability of downward movement (from down market)

#### Common Error Messages

| Message | Cause |
|---------|-------|
| `"No active bitcoin markets found"` | No active up/down markets available |
| `"Market too new. Age: X minutes, need at least 5 minutes"` | Market was created less than 5 minutes ago |
| `"Market already closed at <timestamp>"` | Current time is past the market's closing time |
| `"Insufficient market activity. Up market has X bets, down market has Y bets. Need at least 7 bets on one market and 3 on the other."` | Not enough betting activity |
| `"Up market threshold too close to creation price. Difference: X%, need >= 0.15%"` | Market threshold too close to entry price |
| `"Bias (X%) below threshold (Y%). Signal too weak."` | Bias doesn't meet minimum threshold |
| `"Movement probability (X%) below threshold (Y%). Agents expect flat behavior."` | Movement probability too low |
| `"Divergence detected. Upward bias (X%) not strong enough relative to down market yes votes (Y%). Need consensus, not divergence."` | Consensus requirement not met for long |
| `"Divergence detected. Downward bias (X%) not strong enough relative to up market yes votes (Y%). Need consensus, not divergence."` | Consensus requirement not met for short |

#### Example Requests

**cURL**:
```bash
curl http://agents.eternax.ai/perps_signal
```

**JavaScript (fetch)**:
```javascript
fetch('http://agents.eternax.ai/perps_signal')
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      console.log('No signal:', data.message);
      console.log('Current bias:', data.bias);
    } else {
      console.log('Signal:', data.action);
      console.log('Bias:', data.bias);
      console.log('Reason:', data.reason);
    }
  });
```

**Python (requests)**:
```python
import requests

response = requests.get('http://agents.eternax.ai/perps_signal')
data = response.json()

if 'error' in data:
    print(f"No signal: {data['message']}")
    print(f"Current bias: {data['bias']}")
else:
    print(f"Signal: {data['action']}")
    print(f"Bias: {data['bias']}")
    print(f"Reason: {data['reason']}")
```

#### Example Responses

**Long Signal**:
```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "action": "long",
  "bias": 18.5,
  "close_time": "2024-01-15T11:00:00.000Z",
  "reason": "Strong upward bias (18.5%) with sufficient movement probability (28.3%). Consensus confirmed.",
  "market_id": "market_abc123",
  "price_at_creation": 42150.50
}
```

**Short Signal**:
```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "action": "short",
  "bias": -16.2,
  "close_time": "2024-01-15T11:00:00.000Z",
  "reason": "Strong downward bias (16.2%) with sufficient movement probability (22.1%). Consensus confirmed.",
  "market_id": "market_abc123",
  "price_at_creation": 42150.50
}
```

**No Signal (Weak Bias)**:
```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "error": "No signal available",
  "message": "Bias (8.5%) below threshold (12%). Signal too weak.",
  "action": "no-trade",
  "bias": 8.5,
  "market_id": "market_abc123"
}
```

**No Signal (No Markets)**:
```json
{
  "pair": "BTC-USDC",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "error": "No signal available",
  "message": "No active bitcoin markets found",
  "action": "no-trade",
  "bias": null,
  "market_id": null
}
```

#### Notes

- The endpoint currently only supports Bitcoin (`BTC-USDC`)
- Signals are based on real-time prediction market consensus
- The `bias` field is always included, even in error responses (may be `null` if markets unavailable)
- All timestamps are in ISO 8601 format (UTC)
- The endpoint always returns HTTP 200, even when no signal is available
- Signals are recalculated on each request based on current market state

#### Rate Limiting

No rate limiting is currently implemented. However, signals are based on prediction markets that update every 30 seconds, so polling more frequently than every 30 seconds is unnecessary.

#### Error Handling

If the server encounters an internal error, it will return:

```json
{
  "error": "Failed to generate perps signal",
  "details": "Error message here"
}
```

With HTTP status code `500 Internal Server Error`.

