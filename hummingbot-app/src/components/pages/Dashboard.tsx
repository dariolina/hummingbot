import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useApp } from "@/lib/AppContext";
import { HummingbotAPI, OrderInfo, PositionInfo } from "@/lib/api";
import { formatNumber, formatCurrency, formatDuration } from "@/lib/utils";
import { Activity, DollarSign, TrendingUp, Clock, RefreshCw } from "lucide-react";

export function Dashboard() {
  const { isConnected, strategyStatus, connectors } = useApp();
  const [orders, setOrders] = useState<OrderInfo[]>([]);
  const [positions, setPositions] = useState<PositionInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    if (!isConnected) return;
    setLoading(true);
    try {
      const [ordersData, positionsData] = await Promise.all([
        HummingbotAPI.Trading.listOrders(),
        HummingbotAPI.Trading.listPositions(),
      ]);
      setOrders(ordersData);
      setPositions(positionsData);
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [isConnected]);

  const configuredConnectors = connectors.filter((c) => c.is_configured);
  const totalPnl = positions.reduce(
    (sum, p) => sum + (parseFloat(p.unrealized_pnl || "0")),
    0
  );

  if (!isConnected) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] space-y-4">
        <Activity className="h-16 w-16 text-muted-foreground" />
        <h2 className="text-2xl font-semibold">Not Connected</h2>
        <p className="text-muted-foreground">
          Unable to connect to Hummingbot API. Please check that the bot is running.
        </p>
        <Button onClick={() => window.location.reload()}>
          <RefreshCw className="mr-2 h-4 w-4" />
          Retry Connection
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Strategy Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {strategyStatus?.is_running ? "Running" : "Stopped"}
            </div>
            <p className="text-xs text-muted-foreground">
              {strategyStatus?.strategy_name || "No strategy"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Open Orders</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{orders.length}</div>
            <p className="text-xs text-muted-foreground">
              Across {configuredConnectors.length} connector(s)
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unrealized PnL</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${totalPnl >= 0 ? "text-green-500" : "text-red-500"}`}>
              {formatCurrency(totalPnl)}
            </div>
            <p className="text-xs text-muted-foreground">
              {positions.length} open position(s)
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Runtime</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {strategyStatus?.runtime_seconds
                ? formatDuration(strategyStatus.runtime_seconds)
                : "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">
              {strategyStatus?.strategy_type || ""}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Open Orders */}
      <Card>
        <CardHeader>
          <CardTitle>Open Orders</CardTitle>
          <CardDescription>Currently active orders across all connectors</CardDescription>
        </CardHeader>
        <CardContent>
          {orders.length === 0 ? (
            <p className="text-center text-muted-foreground py-4">No open orders</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Connector</th>
                    <th className="text-left py-2">Pair</th>
                    <th className="text-left py-2">Side</th>
                    <th className="text-right py-2">Price</th>
                    <th className="text-right py-2">Amount</th>
                    <th className="text-right py-2">Filled</th>
                    <th className="text-left py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {orders.map((order) => (
                    <tr key={order.client_order_id} className="border-b">
                      <td className="py-2">{order.connector_name}</td>
                      <td className="py-2">{order.trading_pair}</td>
                      <td className={`py-2 ${order.side === "buy" ? "text-green-500" : "text-red-500"}`}>
                        {order.side.toUpperCase()}
                      </td>
                      <td className="py-2 text-right">{formatNumber(order.price || 0, 4)}</td>
                      <td className="py-2 text-right">{formatNumber(order.amount, 4)}</td>
                      <td className="py-2 text-right">{formatNumber(order.filled_amount, 4)}</td>
                      <td className="py-2">{order.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Open Positions */}
      <Card>
        <CardHeader>
          <CardTitle>Open Positions</CardTitle>
          <CardDescription>Current perpetual positions</CardDescription>
        </CardHeader>
        <CardContent>
          {positions.length === 0 ? (
            <p className="text-center text-muted-foreground py-4">No open positions</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Connector</th>
                    <th className="text-left py-2">Pair</th>
                    <th className="text-left py-2">Side</th>
                    <th className="text-right py-2">Size</th>
                    <th className="text-right py-2">Entry</th>
                    <th className="text-right py-2">PnL</th>
                    <th className="text-right py-2">Leverage</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((pos, idx) => (
                    <tr key={idx} className="border-b">
                      <td className="py-2">{pos.connector_name}</td>
                      <td className="py-2">{pos.trading_pair}</td>
                      <td className={`py-2 ${pos.position_side === "long" ? "text-green-500" : "text-red-500"}`}>
                        {pos.position_side.toUpperCase()}
                      </td>
                      <td className="py-2 text-right">{formatNumber(pos.amount, 4)}</td>
                      <td className="py-2 text-right">{formatNumber(pos.entry_price, 4)}</td>
                      <td className={`py-2 text-right ${parseFloat(pos.unrealized_pnl || "0") >= 0 ? "text-green-500" : "text-red-500"}`}>
                        {formatCurrency(pos.unrealized_pnl || 0)}
                      </td>
                      <td className="py-2 text-right">{pos.leverage || "-"}x</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
