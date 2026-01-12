import asyncio
from typing import Dict, List

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.exchange.paper_trade.trading_pair import TradingPair
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class DynamicOrderbookDemo(ScriptStrategyBase):
    """
    Demo script showing dynamic order book initialization and removal.
    - Starts with BTC-USDT
    - Adds SOL-USDT after 10 seconds
    - Adds ETH-USDT after 15 seconds
    - Removes both SOL-USDT and ETH-USDT after 25 seconds

    Order book data is displayed in format_status() (use `status` command to view).

    Usage: start --script dynamic_orderbook_demo.py
    """

    markets = {"binance": {"BTC-USDT"}}

    # Configuration
    ORDER_BOOK_DEPTH = 5  # Number of levels to display
    HISTOGRAM_RANGE_BPS = 50  # ±50 bps from mid price
    CHART_HEIGHT = 12  # Height of depth chart in rows

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self._start_timestamp = None
        self._sol_added = False
        self._eth_added = False
        self._sol_removed = False
        self._eth_removed = False
        self._market_data_provider = MarketDataProvider(connectors)

    def on_tick(self):
        if self._start_timestamp is None:
            self._start_timestamp = self.current_timestamp

        elapsed = self.current_timestamp - self._start_timestamp

        # Add SOL-USDT after 10 seconds
        if elapsed >= 10 and not self._sol_added:
            self._sol_added = True
            self.logger().info(">>> ADDING SOL-USDT ORDER BOOK <<<")
            asyncio.create_task(self._add_trading_pair("SOL-USDT"))

        # Add ETH-USDT after 15 seconds
        if elapsed >= 15 and not self._eth_added:
            self._eth_added = True
            self.logger().info(">>> ADDING ETH-USDT ORDER BOOK <<<")
            asyncio.create_task(self._add_trading_pair("ETH-USDT"))

        # Remove both SOL-USDT and ETH-USDT after 25 seconds
        if elapsed >= 25 and not self._sol_removed and self._sol_added:
            self._sol_removed = True
            self.logger().info(">>> REMOVING SOL-USDT ORDER BOOK <<<")
            asyncio.create_task(self._remove_trading_pair("SOL-USDT"))

        if elapsed >= 25 and not self._eth_removed and self._eth_added:
            self._eth_removed = True
            self.logger().info(">>> REMOVING ETH-USDT ORDER BOOK <<<")
            asyncio.create_task(self._remove_trading_pair("ETH-USDT"))

    def format_status(self) -> str:
        """
        Displays order book information for all tracked trading pairs.
        Use the `status` command to view this output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []
        connector = self.connectors["binance"]
        elapsed = self.current_timestamp - self._start_timestamp if self._start_timestamp else 0

        lines.append("\n" + "=" * 80)
        lines.append(f"  DYNAMIC ORDER BOOK DEMO | Elapsed: {elapsed:.1f}s")
        lines.append("=" * 80)

        # Timeline status
        lines.append("\n  Timeline:")
        lines.append(
            f"    [{'✓' if self._sol_added else '○'}] 10s - Add SOL-USDT" + (" (added)" if self._sol_added else ""))
        lines.append(
            f"    [{'✓' if self._eth_added else '○'}] 15s - Add ETH-USDT" + (" (added)" if self._eth_added else ""))
        lines.append(f"    [{'✓' if self._sol_removed else '○'}] 25s - Remove SOL-USDT & ETH-USDT" + (
            " (removed)" if self._sol_removed else ""))

        # Get tracked pairs from order book tracker
        tracker = connector.order_book_tracker
        tracked_pairs = list(tracker.order_books.keys())
        lines.append(f"\n  Tracked Pairs: {tracked_pairs}")

        # Display order book for each pair
        for pair in tracked_pairs:
            lines.append("\n" + "-" * 80)
            lines.extend(self._format_order_book(connector, pair))

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def _format_order_book(self, connector, trading_pair: str) -> List[str]:
        """Format order book data for a single trading pair with horizontal depth chart."""
        lines = []

        try:
            ob = connector.order_book_tracker.order_books.get(trading_pair)
            if ob is None:
                lines.append(f"  {trading_pair}: Order book not yet initialized...")
                return lines

            bids_df, asks_df = ob.snapshot

            if len(bids_df) == 0 or len(asks_df) == 0:
                lines.append(f"  {trading_pair}: Order book empty or initializing...")
                return lines

            # Calculate market metrics
            best_bid = float(bids_df.iloc[0].price)
            best_ask = float(asks_df.iloc[0].price)
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            mid_price = (best_bid + best_ask) / 2

            # Header with market info
            lines.append(f"  {trading_pair}")
            lines.append(f"  Mid: {mid_price:.4f} | Spread: {spread:.4f} ({spread_pct:.4f}%)")
            lines.append("")

            # Prepare order book display
            depth = min(self.ORDER_BOOK_DEPTH, len(bids_df), len(asks_df))

            # Order book table
            lines.append(f"  {'Bid Size':>12} {'Bid Price':>14} │ {'Ask Price':<14} {'Ask Size':<12}")
            lines.append(f"  {'-' * 12} {'-' * 14} │ {'-' * 14} {'-' * 12}")

            for i in range(depth):
                bid_price = float(bids_df.iloc[i].price)
                bid_size = float(bids_df.iloc[i].amount)
                ask_price = float(asks_df.iloc[i].price)
                ask_size = float(asks_df.iloc[i].amount)
                lines.append(f"  {bid_size:>12.4f} {bid_price:>14.4f} │ {ask_price:<14.4f} {ask_size:<12.4f}")

            # Total volume at displayed levels
            total_bid_vol = float(bids_df.iloc[:depth]['amount'].sum())
            total_ask_vol = float(asks_df.iloc[:depth]['amount'].sum())
            lines.append(f"  {'-' * 12} {'-' * 14} │ {'-' * 14} {'-' * 12}")
            lines.append(f"  {'Total:':>12} {total_bid_vol:>14.4f} │ {total_ask_vol:<14.4f}")

            # Add horizontal depth chart below
            lines.extend(self._build_horizontal_depth_chart(bids_df, asks_df, mid_price))

        except Exception as e:
            lines.append(f"  {trading_pair}: Error - {e}")

        return lines

    def _build_horizontal_depth_chart(self, bids_df, asks_df, mid_price: float) -> List[str]:
        """
        Build ASCII depth chart with bps on X-axis and volume on Y-axis.
        1 bar per bps, bids on left, asks on right.
        """
        lines = []
        num_buckets = self.HISTOGRAM_RANGE_BPS  # 1 bucket per bps
        range_decimal = self.HISTOGRAM_RANGE_BPS / 10000
        bucket_size_decimal = range_decimal / num_buckets

        # Aggregate bid volume into buckets (1 per bps, from mid going down)
        bid_buckets = []
        for i in range(num_buckets):
            bucket_upper = mid_price * (1 - i * bucket_size_decimal)
            bucket_lower = mid_price * (1 - (i + 1) * bucket_size_decimal)
            mask = (bids_df['price'] <= bucket_upper) & (bids_df['price'] > bucket_lower)
            vol = float(bids_df[mask]['amount'].sum()) if mask.any() else 0
            bid_buckets.append(vol)

        # Aggregate ask volume into buckets (1 per bps, from mid going up)
        ask_buckets = []
        for i in range(num_buckets):
            bucket_lower = mid_price * (1 + i * bucket_size_decimal)
            bucket_upper = mid_price * (1 + (i + 1) * bucket_size_decimal)
            mask = (asks_df['price'] >= bucket_lower) & (asks_df['price'] < bucket_upper)
            vol = float(asks_df[mask]['amount'].sum()) if mask.any() else 0
            ask_buckets.append(vol)

        # Find max volume for scaling
        all_vols = bid_buckets + ask_buckets
        max_vol = max(all_vols) if all_vols and max(all_vols) > 0 else 1

        # Calculate totals and imbalance
        total_bid_vol = sum(bid_buckets)
        total_ask_vol = sum(ask_buckets)
        total_vol = total_bid_vol + total_ask_vol
        imbalance_pct = ((total_bid_vol - total_ask_vol) / total_vol * 100) if total_vol > 0 else 0
        imbalance_str = f"+{imbalance_pct:.1f}%" if imbalance_pct >= 0 else f"{imbalance_pct:.1f}%"

        # Calculate bar heights
        chart_height = self.CHART_HEIGHT
        bid_heights = [int((v / max_vol) * chart_height) for v in bid_buckets]
        ask_heights = [int((v / max_vol) * chart_height) for v in ask_buckets]

        # Reverse bids so furthest from mid is on left
        bid_heights_display = list(reversed(bid_heights))

        # Header with summary
        lines.append("")
        lines.append(f"  Depth Chart (±{self.HISTOGRAM_RANGE_BPS} bps)")
        lines.append(f"  Bids: {total_bid_vol:,.2f}  |  Asks: {total_ask_vol:,.2f}  |  Imbalance: {imbalance_str}")
        lines.append("")

        # Build chart row by row from top to bottom
        for row in range(chart_height, 0, -1):
            row_chars = []

            # Bid bars (left side)
            for h in bid_heights_display:
                if h >= row:
                    row_chars.append("█")
                else:
                    row_chars.append(" ")

            # Center divider
            row_chars.append("│")

            # Ask bars (right side)
            for h in ask_heights:
                if h >= row:
                    row_chars.append("█")
                else:
                    row_chars.append(" ")

            lines.append("  " + "".join(row_chars))

        # X-axis line
        lines.append("  " + "─" * num_buckets + "┴" + "─" * num_buckets)

        # X-axis labels (sparse - every 10 bps)
        label_interval = 10
        # Build label line with proper spacing
        bid_label_line = [" "] * num_buckets
        ask_label_line = [" "] * num_buckets

        # Place bid labels (negative values, from left to right: -50, -40, -30, -20, -10)
        for bps in range(self.HISTOGRAM_RANGE_BPS, 0, -label_interval):
            pos = self.HISTOGRAM_RANGE_BPS - bps  # Position from left
            label = f"-{bps}"
            # Center the label at this position
            start = max(0, pos - len(label) // 2)
            for j, ch in enumerate(label):
                if start + j < num_buckets:
                    bid_label_line[start + j] = ch

        # Place ask labels (positive values, from left to right: 10, 20, 30, 40, 50)
        for bps in range(label_interval, self.HISTOGRAM_RANGE_BPS + 1, label_interval):
            pos = bps - 1  # Position from left (0-indexed)
            label = f"+{bps}"
            start = max(0, pos - len(label) // 2)
            for j, ch in enumerate(label):
                if start + j < num_buckets:
                    ask_label_line[start + j] = ch

        lines.append("  " + "".join(bid_label_line) + "0" + "".join(ask_label_line) + "  (bps)")

        return lines

    async def _add_trading_pair(self, trading_pair: str):
        try:
            success = await self._market_data_provider.initialize_order_book("binance", trading_pair)
            if not success:
                self.logger().error(f"Failed to add {trading_pair} to order book tracker")
                return

            # For paper_trade connector, also add to _trading_pairs dict
            connector = self.connectors["binance"]
            if hasattr(connector, '_trading_pairs') and isinstance(connector._trading_pairs, dict):
                base, quote = trading_pair.split("-")
                exchange_pair = f"{base}{quote}"
                connector._trading_pairs[trading_pair] = TradingPair(
                    trading_pair=exchange_pair,
                    base_asset=base,
                    quote_asset=quote
                )

            self.logger().info(f"Successfully added {trading_pair}!")

        except Exception as e:
            self.logger().exception(f"Error adding {trading_pair}: {e}")

    async def _remove_trading_pair(self, trading_pair: str):
        try:
            success = await self._market_data_provider.remove_order_book("binance", trading_pair)
            if not success:
                self.logger().error(f"Failed to remove {trading_pair} from order book tracker")
                return

            # For paper_trade connector, also remove from _trading_pairs dict
            connector = self.connectors["binance"]
            if hasattr(connector, '_trading_pairs') and isinstance(connector._trading_pairs, dict):
                connector._trading_pairs.pop(trading_pair, None)

            self.logger().info(f"Successfully removed {trading_pair}!")

        except Exception as e:
            self.logger().exception(f"Error removing {trading_pair}: {e}")
