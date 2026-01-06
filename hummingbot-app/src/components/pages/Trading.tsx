import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useApp } from "@/lib/AppContext";
import { HummingbotAPI, OrderInfo, MarketPrice, PlaceOrderRequest } from "@/lib/api";
import { formatNumber } from "@/lib/utils";
import { RefreshCw, TrendingUp, TrendingDown, X } from "lucide-react";

export function Trading() {
  const { isConnected, connectors, selectedConnector, setSelectedConnector } = useApp();
  const [tradingPair, setTradingPair] = useState("");
  const [orders, setOrders] = useState<OrderInfo[]>([]);
  const [price, setPrice] = useState<MarketPrice | null>(null);
  const [loading, setLoading] = useState(false);

  // Order form state
  const [orderSide, setOrderSide] = useState<"buy" | "sell">("buy");
  const [orderType, setOrderType] = useState<"limit" | "market">("limit");
  const [orderAmount, setOrderAmount] = useState("");
  const [orderPrice, setOrderPrice] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const configuredConnectors = connectors.filter((c) => c.is_configured);

  const fetchOrders = async () => {
    if (!selectedConnector) return;
    setLoading(true);
    try {
      const ordersData = await HummingbotAPI.Trading.listOrders(selectedConnector, tradingPair || undefined);
      setOrders(ordersData);
    } catch (error) {
      console.error("Failed to fetch orders:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPrice = async () => {
    if (!selectedConnector || !tradingPair) return;
    try {
      const priceData = await HummingbotAPI.Trading.getPrice(selectedConnector, tradingPair);
      setPrice(priceData);
    } catch (error) {
      console.error("Failed to fetch price:", error);
    }
  };

  useEffect(() => {
    if (selectedConnector) {
      fetchOrders();
    }
  }, [selectedConnector, tradingPair]);

  useEffect(() => {
    if (selectedConnector && tradingPair) {
      fetchPrice();
      const interval = setInterval(fetchPrice, 5000);
      return () => clearInterval(interval);
    }
  }, [selectedConnector, tradingPair]);

  const handlePlaceOrder = async () => {
    if (!selectedConnector || !tradingPair || !orderAmount) return;
    if (orderType === "limit" && !orderPrice) return;

    setSubmitting(true);
    try {
      const request: PlaceOrderRequest = {
        connector_name: selectedConnector,
        trading_pair: tradingPair,
        side: orderSide,
        order_type: orderType,
        amount: orderAmount,
        price: orderType === "limit" ? orderPrice : undefined,
      };
      await HummingbotAPI.Trading.placeOrder(request);
      setOrderAmount("");
      setOrderPrice("");
      await fetchOrders();
    } catch (error) {
      console.error("Failed to place order:", error);
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancelOrder = async (order: OrderInfo) => {
    try {
      await HummingbotAPI.Trading.cancelOrder(
        order.client_order_id,
        order.connector_name,
        order.trading_pair
      );
      await fetchOrders();
    } catch (error) {
      console.error("Failed to cancel order:", error);
    }
  };

  if (!isConnected) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] space-y-4">
        <TrendingUp className="h-16 w-16 text-muted-foreground" />
        <h2 className="text-2xl font-semibold">Not Connected</h2>
        <p className="text-muted-foreground">Connect to Hummingbot to start trading</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Trading</h1>
          <p className="text-muted-foreground">Place and manage orders</p>
        </div>
        <Button variant="outline" size="sm" onClick={fetchOrders} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Trading Form */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Place Order</CardTitle>
            <CardDescription>Create a new order on the selected connector</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Connector Selection */}
              <div className="space-y-2">
                <Label>Connector</Label>
                <select
                  className="w-full h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={selectedConnector || ""}
                  onChange={(e) => setSelectedConnector(e.target.value || null)}
                >
                  <option value="">Select connector</option>
                  {configuredConnectors.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.display_name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Trading Pair */}
              <div className="space-y-2">
                <Label htmlFor="trading_pair">Trading Pair</Label>
                <Input
                  id="trading_pair"
                  placeholder="e.g., BTC-USDT"
                  value={tradingPair}
                  onChange={(e) => setTradingPair(e.target.value.toUpperCase())}
                />
              </div>

              {/* Current Price */}
              {price && (
                <div className="p-3 rounded-lg bg-muted">
                  <div className="text-sm text-muted-foreground">Current Price</div>
                  <div className="text-2xl font-bold">{formatNumber(price.mid_price, 4)}</div>
                  <div className="flex justify-between text-sm">
                    <span className="text-green-500">Bid: {formatNumber(price.bid_price || 0, 4)}</span>
                    <span className="text-red-500">Ask: {formatNumber(price.ask_price || 0, 4)}</span>
                  </div>
                </div>
              )}

              {/* Side Selection */}
              <div className="space-y-2">
                <Label>Side</Label>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant={orderSide === "buy" ? "default" : "outline"}
                    className={orderSide === "buy" ? "bg-green-600 hover:bg-green-700" : ""}
                    onClick={() => setOrderSide("buy")}
                  >
                    <TrendingUp className="mr-2 h-4 w-4" />
                    Buy
                  </Button>
                  <Button
                    variant={orderSide === "sell" ? "default" : "outline"}
                    className={orderSide === "sell" ? "bg-red-600 hover:bg-red-700" : ""}
                    onClick={() => setOrderSide("sell")}
                  >
                    <TrendingDown className="mr-2 h-4 w-4" />
                    Sell
                  </Button>
                </div>
              </div>

              {/* Order Type */}
              <div className="space-y-2">
                <Label>Order Type</Label>
                <Tabs value={orderType} onValueChange={(v) => setOrderType(v as "limit" | "market")}>
                  <TabsList className="w-full">
                    <TabsTrigger value="limit" className="flex-1">Limit</TabsTrigger>
                    <TabsTrigger value="market" className="flex-1">Market</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>

              {/* Price (for limit orders) */}
              {orderType === "limit" && (
                <div className="space-y-2">
                  <Label htmlFor="order_price">Price</Label>
                  <Input
                    id="order_price"
                    type="number"
                    placeholder="0.00"
                    value={orderPrice}
                    onChange={(e) => setOrderPrice(e.target.value)}
                  />
                </div>
              )}

              {/* Amount */}
              <div className="space-y-2">
                <Label htmlFor="order_amount">Amount</Label>
                <Input
                  id="order_amount"
                  type="number"
                  placeholder="0.00"
                  value={orderAmount}
                  onChange={(e) => setOrderAmount(e.target.value)}
                />
              </div>

              {/* Submit Button */}
              <Button
                className={`w-full ${orderSide === "buy" ? "bg-green-600 hover:bg-green-700" : "bg-red-600 hover:bg-red-700"}`}
                onClick={handlePlaceOrder}
                disabled={submitting || !selectedConnector || !tradingPair || !orderAmount || (orderType === "limit" && !orderPrice)}
              >
                {submitting ? "Placing Order..." : `${orderSide === "buy" ? "Buy" : "Sell"} ${tradingPair || "..."}`}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Orders List */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Open Orders</CardTitle>
            <CardDescription>Your active orders on {selectedConnector || "all connectors"}</CardDescription>
          </CardHeader>
          <CardContent>
            {orders.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No open orders</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Pair</th>
                      <th className="text-left py-2">Type</th>
                      <th className="text-left py-2">Side</th>
                      <th className="text-right py-2">Price</th>
                      <th className="text-right py-2">Amount</th>
                      <th className="text-right py-2">Filled</th>
                      <th className="text-right py-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orders.map((order) => (
                      <tr key={order.client_order_id} className="border-b">
                        <td className="py-2">{order.trading_pair}</td>
                        <td className="py-2">{order.order_type}</td>
                        <td className={`py-2 ${order.side === "buy" ? "text-green-500" : "text-red-500"}`}>
                          {order.side.toUpperCase()}
                        </td>
                        <td className="py-2 text-right">{formatNumber(order.price || 0, 4)}</td>
                        <td className="py-2 text-right">{formatNumber(order.amount, 4)}</td>
                        <td className="py-2 text-right">{formatNumber(order.filled_amount, 4)}</td>
                        <td className="py-2 text-right">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleCancelOrder(order)}
                          >
                            <X className="h-4 w-4 text-destructive" />
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
