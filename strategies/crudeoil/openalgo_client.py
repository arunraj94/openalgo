# openalgo_client.py
# Thin wrapper over OpenAlgo API + WebSocket for Unified Straddle Bot
# Ensures consistent event routing and mandatory printing of ticks.

import threading
import asyncio
import traceback
from openalgo import api

class OpenAlgoClientWrapper:
    """
    Wrapper around OpenAlgo REST + WebSocket.
    Provides:
      - connect()
      - subscribe_ltp()
      - unified on_ltp callback routing
      - safe shutdown
    """

    def __init__(self, api_key, host, ws_url, on_ltp_callback):
        self.api_key = api_key
        self.host = host
        self.ws_url = ws_url
        self.on_ltp_callback = on_ltp_callback

        self.client = api(api_key=api_key, host=host, ws_url=ws_url)
        self.connected = False
        self.ws_thread = None
        self.orderbook = None
        self.future_symbol = None
        self.nearest_expiry = None
        self.symbol = None
        self.exchange = None
        self.strike_step = 50

    def connect(self):
        try:
            self.client.connect()
            self.connected = True
            print("[OpenAlgoClient] Connected to REST + WS server.")
        except Exception:
            print("[OpenAlgoClient] ERROR: Failed to connect.")
            traceback.print_exc()
            self.connected = False

    def subscribe_ltp(self, instruments):
        if not self.connected:
            print("[OpenAlgoClient] Cannot subscribe. Not connected.")
            return
        try:
            self.client.subscribe_ltp(instruments, on_data_received=self.on_ltp_callback)
            print(f"[OpenAlgoClient] Subscribed LTP: {instruments}")
        except Exception:
            print("[OpenAlgoClient] Subscribe LTP failed.")
            traceback.print_exc()

    def subscribe_depth(self, instruments):
        if not self.connected:
            print("[OpenAlgoClient] Cannot subscribe. Not connected.")
            return
        try:
            self.client.subscribe_depth(instruments)
            print(f"[OpenAlgoClient] Subscribed LTP: {instruments}")
        except Exception:
            print("[OpenAlgoClient] Subscribe LTP failed.")
            traceback.print_exc()


    def place_order(self, **kwargs):
        """Safe wrapper for REST order placement."""
        try:
            return self.client.optionsorder(**kwargs)
        except Exception:
            print("[OpenAlgoClient] ERROR placing order.")
            traceback.print_exc()
            return None

    def disconnect(self):
        try:
            self.client.disconnect()
        except Exception:
            pass
        print("[OpenAlgoClient] Disconnected.")

    def get_nearest_future_symbol(self):
        """
        Get the nearest future symbol for the given MCX commodity (e.g. CRUDEOILM).
        Ensures the symbol follows OpenAlgo format without hyphens.
        Example Output: CRUDEOILM19NOV25FUT
        """
        try:
           
            expiry_list = self.client.expiry(
                symbol=self.symbol,
                exchange=self.exchange,
                instrumenttype="futures"
            )

            if not expiry_list or 'data' not in expiry_list or not expiry_list['data']:
                print("No future expiry data received from client API.")
                return None

            # Take the first (nearest) expiry
            nearest_future_expiry = expiry_list['data'][0]

            # Remove hyphens if present (e.g., "19-NOV-25" -> "19NOV25")
            clean_expiry = nearest_future_expiry.replace("-", "")

            # Build valid OpenAlgo MCX future symbol
            fut_symbol = f"{self.symbol.upper()}{clean_expiry}FUT"

            print(f"Nearest Future Symbol: {fut_symbol}")
            return fut_symbol

        except Exception as e:
            print(f"[ERROR] Failed to fetch nearest future: {e}")
            return None


    def get_nearest_expiry(self):
        """
        Get the nearest option expiry for CRUDEOILM on MCX.
        Example Output: 20NOV25
        """
        try:
            expiry_dates = self.client.expiry(
                symbol=self.symbol,
                exchange=self.exchange,
                instrumenttype='options'
            )
            if not expiry_dates or 'data' not in expiry_dates or not expiry_dates['data']:
                print("No option expiry data received from client API.")
                return None
            nearest_expiry = expiry_dates['data'][0]
            print(f"Nearest Option Expiry: {nearest_expiry}")
            self.nearest_expiry = nearest_expiry
            return nearest_expiry
        except Exception as e:
            print(f"[ERROR] Failed to fetch nearest option expiry: {e}")
            raise

    def get_option_symbols(self, symbol, exchange):
        """
        Derive ATM strike and generate CE/PE option symbols for CRUDEOILM (MCX).
        Automatically removes hyphens from expiry to match OpenAlgo symbol format.
        """
        # Step 1: Get nearest future symbol
        self.symbol = symbol
        self.exchange = exchange
        future_symbol = self.get_nearest_future_symbol()
        if not future_symbol:
            return None, None

        # Step 2: Fetch quote for nearest future
        quote = self.client.quotes(symbol=future_symbol, exchange=self.exchange)
        print(f"The Quote for {future_symbol} is {quote}")
        price = quote["data"]["ltp"]
        if not price:
            print("The price is not updated in the QUOTE")
            return None, None

        # Step 3: Determine ATM strike (rounded)
        atm_price = self.get_ATM_strike(price)

        # Step 4: Get nearest expiry (and clean format)
        nearest_expiry = self.get_nearest_expiry()
        if not nearest_expiry:
            return None, None
        clean_expiry = nearest_expiry.replace("-", "")  # e.g. "17-NOV-25" → "17NOV25"

        # Step 5: Define strike range
        strike_range = [atm_price + (i * self.strike_step) for i in range(-20, 21)]

        # Step 6: Build CE and PE symbols (using build_option_symbol)
        ce_symbols = [
            self.build_option_symbol(self.symbol, nearest_expiry, strike, "CE")
        for strike in strike_range
        ]
        pe_symbols = [
            self.build_option_symbol(self.symbol, nearest_expiry, strike, "PE")
            for strike in strike_range
        ]

        print(f"ATM Price: {atm_price}")
        print(f"CE Symbols: {ce_symbols[:3]} ...")
        print(f"PE Symbols: {pe_symbols[:3]} ...")

        index = [{'exchange': 'MCX', 'symbol': future_symbol}]
        self.future_symbol = future_symbol

        return ce_symbols + pe_symbols + index

    def get_ATM_strike(self, price: float) -> int:
        """
        Round the underlying price to the nearest strike step to find the ATM strike.

        Args:
            price (float): The underlying price.

        Returns:
            int: The ATM strike, rounded to the nearest STRIKE_STEP.
        """
        return int(round(price / self.strike_step) * self.strike_step)

    def build_option_symbol(self, base_symbol, expiry, strike, option_type):
        """
        Build valid MCX option symbol for OpenAlgo.
        Example Output: {'exchange': 'MCX', 'symbol': 'CRUDEOILM17NOV255350CE'}
        """
        # Ensure expiry format is clean (e.g., '17-NOV-25' -> '17NOV25')
        expiry_fmt = expiry.replace("-", "").upper()

        # Format strike (MCX usually integers like 5400, 5450)
        strike_int = int(strike)

        # Build OpenAlgo-compatible symbol dictionary
        symbol = f"{base_symbol.upper()}{expiry_fmt}{strike_int}{option_type.upper()}"

        return {
            "exchange": "MCX",
            "symbol": symbol
        }

    def get_exchange_ltp(self):
        if not self.exchange or not self.future_symbol:
            raise ValueError("Exchange or future symbol not set. Ensure get_option_symbols() was called.")
        try:
            ltp_data = self.client.get_ltp(exchange=self.exchange, symbol=self.future_symbol)
            if ltp_data and 'ltp' in ltp_data:
                return ltp_data['ltp'][self.exchange][self.future_symbol]['ltp']
        except Exception as e:
            print(f"[OpenAlgoClient] Error fetching exchange LTP: {e}")
        return 0.0

    def get_order_info_of_order(self, order_id):
        # Get latest orderbook from centralized feed (fallback to self.orderbook if feed not available)
        orderbook = self.orderbook
        if not orderbook:
            try:
                orderbook = self.client.orderbook()
                self.orderbook = orderbook
            except Exception as err:
                print(f"[OpenAlgoClient] Failed to fetch orderbook: {err}")
                return None
        data = orderbook.get('data', {})
        orders = data.get('orders', [])
        for order in orders:
            if order.get('orderid') == order_id:
                return order
        return None

    def get_market_depth(self, symbol):
        """
        Fetches market depth for a symbol and returns a standardized dict.
        Returns: {'bids': [{'price': float, 'quantity': int}], 'asks': ...}
        """
        try:
            depth_data = self.client.get_depth(symbol=symbol, exchange=self.exchange)
            # Structure: {'status': 'success', 'depth': {'MCX': {'SYMBOL': {'buyBook': {...}, 'sellBook': {...}}}}}
            
            if not depth_data or 'depth' not in depth_data:
                return None
                
            exch_data = depth_data['depth'].get(self.exchange, {})
            sym_data = exch_data.get(symbol, {})
            
            buy_book = sym_data.get('buyBook', {})
            sell_book = sym_data.get('sellBook', {})
            
            bids = []
            asks = []
            
            # OpenAlgo returns dict keys '1', '2', '3', '4', '5'
            for i in range(1, 6):
                bid_node = buy_book.get(str(i))
                if bid_node:
                    bids.append({'price': float(bid_node.get('price', 0)), 'quantity': int(bid_node.get('quantity', 0))})
                    
                ask_node = sell_book.get(str(i))
                if ask_node:
                    asks.append({'price': float(ask_node.get('price', 0)), 'quantity': int(ask_node.get('quantity', 0))})
            
            return {'bids': bids, 'asks': asks}
            
        except Exception as e:
            print(f"[OpenAlgoClient] Error fetching depth for {symbol}: {e}")
            return None

    def get_quote(self, symbol):
        """
        Fetches quote for a symbol.
        Returns: {'ltp': float, 'ask': float, 'bid': float, ...}
        """
        try:
            quote_data = self.client.quotes(symbol=symbol, exchange=self.exchange)
            # Structure: {'status': 'success', 'data': {'symbol': ..., 'ltp': ..., 'ask': ..., 'bid': ...}}
            if not quote_data or 'data' not in quote_data:
                return None
            return quote_data['data']
        except Exception as e:
            print(f"[OpenAlgoClient] Error fetching quote for {symbol}: {e}")
            return None

    async def async_place_orders(self, symbol, order_type, quantity, strategy_tag='CRUDEOIL_STRATEGY1'):
        """
        Places an Limit order and if not executed, retry the same
        """
        status = 'open'
        ltp = 0
        try:
            depth = self.client.get_depth(symbol=symbol, exchange=self.exchange)
            exch_data = depth.get('depth', {}).get(self.exchange, {}).get(symbol, {})
            
            if order_type == 'BUY':
                # Buy at Best Ask
                ltp = exch_data.get('sellBook', {}).get('1', {}).get('price', 0)
            else:
                # Sell at Best Bid
                ltp = exch_data.get('buyBook', {}).get('1', {}).get('price', 0)
        except Exception:
            print(f"Error fetching depth for placement. Using 0.")
            ltp = 0

        print(f"Placing {order_type} for {symbol} at {ltp}")
        
        order_response = await self.client.placeorder_async(
            strategy=strategy_tag,
            symbol=symbol,
            action=order_type,
            exchange=self.exchange,
            price_type="LIMIT",
            product='NRML',
            quantity=quantity,
            price=ltp,
        )

        if not order_response or order_response.get('status') == 'error':
            print(f"[OpenAlgoClient] Order Placement Failed: {order_response}")
            return None

        for i in range(0, 5):
            await asyncio.sleep(0.4)
            order = self.get_order_info_of_order(order_response['orderid'])
            if order:
                status = order['order_status']
                if status != 'open':
                    break
                if order_type == 'BUY':
                    ltp = self.client.get_depth(symbol=symbol, exchange=self.exchange)['depth'][self.exchange][symbol]['sellBook']['1']['price']
                else:
                    ltp = self.client.get_depth(symbol=symbol, exchange=self.exchange)['depth'][self.exchange][symbol]['buyBook']['1']['price']
                # ltp = self.client.get_ltp(exchange=self.options_exchange, symbol=symbol)['ltp']['NFO'][symbol]['ltp']
                retry_order = await self.client.modifyorder_async(
                    strategy=strategy_tag,
                    order_id=order_response['orderid'],
                    symbol=symbol,
                    action=order_type,
                    exchange=self.exchange,
                    price_type="LIMIT",
                    product='NRML',
                    quantity=quantity,
                    price=ltp,
                )
                if retry_order['status'] == 'error':
                    if retry_order['message'] == 'Cannot modify order in complete status':
                        status = 'completed'
                    # Order is already executed or rejected
                    break
        if status == 'open':
            print("The order status is open after 5 retries, hence executing the market order")
            cancel_order = await self.client.cancelorder_async(order_id=order_response['orderid'], strategy=strategy_tag)
            if cancel_order['status'] != 'error':
                print(f"Placing {order_type} for {symbol} at market price")            
                order_response = await self.client.placeorder_async(
                    strategy=strategy_tag,
                    symbol=symbol,
                    action=order_type,
                    exchange=self.exchange,
                    price_type="MARKET",
                    product='NRML',
                    quantity=quantity
                )
            await asyncio.sleep(0.4)
        return self.get_order_info_of_order(order_response['orderid'])

    async def modify_sl_to_buycost(self, leg, strategy_tag='CRUDEOIL_STRATEGY1'):
        symbol = leg['sl_order']['symbol']
        symbol = leg['sl_order']['symbol']
        current_ltp = 0.0
        try:
            ltp_data = self.client.get_ltp(exchange=self.exchange, symbol=symbol)
            if ltp_data and 'ltp' in ltp_data:
                current_ltp = float(ltp_data['ltp'][self.exchange][symbol]['ltp'])
        except Exception:
            pass
            
        if current_ltp < leg['sell_price']:
            await self.client.modifyorder_async(
                strategy=strategy_tag,
                order_id=leg['sl_order']['orderid'],
                symbol=leg['sl_order']['symbol'],
                action='BUY',
                exchange=self.exchange,
                price_type="SL",
                product='NRML',
                quantity=leg['sl_order']['quantity'],
                price=round(((5 / 100.0) + 1) * leg['sell_price'], 1),
                trigger_price=leg['sell_price']
            )
        else:
            order = await self.async_modify_orders_to_exit(leg['sl_order']['symbol'], 'BUY',
                                                            leg['sl_order']['quantity'], leg['sl_order']['orderid'], strategy_tag)
            leg['status'] = 'CLOSED'
        await asyncio.sleep(0.4)
    
    async def modify_sl_to_cost(self, leg, cost, strategy_tag='CRUDEOIL_STRATEGY1'):
        symbol = leg['sl_order']['symbol']
        if self.client.get_ltp(exchange=self.exchange, symbol=symbol)['ltp'][self.exchange][symbol]['ltp'] < leg['sell_price']:
            await self.client.modifyorder_async(
                strategy=strategy_tag,
                order_id=leg['sl_order']['orderid'],
                symbol=leg['sl_order']['symbol'],
                action='BUY',
                exchange=self.exchange,
                price_type="SL",
                product='NRML',
                quantity=leg['sl_order']['quantity'],
                price=round(((5 / 100.0) + 1) * leg['sell_price'], 1),
                trigger_price=cost
            )

        else:
            order = await self.async_modify_orders_to_exit(leg['sl_order']['symbol'], 'BUY',
                                                            leg['sl_order']['quantity'], leg['sl_order']['orderid'], strategy_tag)
        await asyncio.sleep(0.4)

    async def async_modify_orders_to_exit(self, symbol, order_type, quantity, orderid, strategy_tag='CRUDEOIL_STRATEGY1'):
        """
        Modify an Limit order and if not executed, retry the same

        Args:
            symbol (str): The trading symbol for the order.
            order_type (str): The action for the order (e.g., 'BUY' or 'SELL').

        Returns:
            dict: The final status or response of the order.
        """
        status = 'open'
        for i in range(0, 5):
            await asyncio.sleep(0.4)
            order = self.get_order_info_of_order(orderid)
            if order:
                status = order['order_status']
                if status != 'open':
                    break
                if order_type == 'SELL':
                    ltp = self.client.get_depth(symbol=symbol, exchange=self.exchange)['depth'][self.exchange][symbol]['buyBook']['1']['price']
                else:
                    ltp = self.client.get_depth(symbol=symbol, exchange=self.exchange)['depth'][self.exchange][symbol]['sellBook']['1']['price']

                print(ltp)
                # ltp = self.client.get_ltp(exchange=self.options_exchange, symbol=symbol)['ltp']['NFO'][symbol]['ltp']
                retry_order = await self.client.modifyorder_async(
                    strategy=strategy_tag,
                    order_id=orderid,
                    symbol=symbol,
                    action=order_type,
                    exchange=self.exchange,
                    price_type="LIMIT",
                    product='NRML',
                    quantity=quantity,
                    price=ltp,
                )
                if retry_order['status'] == 'error':
                   status = 'completed'
                   break
        if status == 'open':     
            print("The order status is open after 5 retries, hence executing the market order")
            cancel_order = await self.client.cancelorder_async(order_id=orderid, strategy=strategy_tag)
            if cancel_order['status'] != 'error':               
                order_response = await self.client.placeorder_async(
                    strategy=strategy_tag,
                    symbol=symbol,
                    action=order_type,
                    exchange=self.exchange,
                    price_type="MARKET",
                    product='NRML',
                    quantity=quantity
                )
                return self.get_order_info_of_order(order_response['orderid'])
        return self.get_order_info_of_order(orderid)

    

    async def async_sl_order(self, order_info, sl_percentage, strategy_tag='CRUDEOIL_STRATEGY1'):
        """
        Places a Stop Loss order based on the primary order.
        Handles both BUY (Long) and SELL (Short) primary orders.
        """
        order_id = order_info['orderid']
        order_status = order_info['order_status']
        print(f"Placing SL for {order_id}, Status: {order_status}")
        
        if order_status != 'complete':
            # No need to set SL for Non completed order
            return

        primary_action = order_info.get('action') or order_info.get('transaction_type') # BUY or SELL
        symbol = order_info['symbol']
        qty = order_info['quantity']
        avg_price = float(order_info.get('average_price', order_info.get('price', 0)))
        
        sl_action = 'SELL' if primary_action == 'BUY' else 'BUY'
        
        # Calculate SL Price
        # Calculate SL Price
        current_ltp = 0.0
        try:
            ltp_data = self.client.get_ltp(exchange=self.exchange, symbol=symbol)
            if ltp_data and 'ltp' in ltp_data:
                current_ltp = float(ltp_data['ltp'][self.exchange][symbol]['ltp'])
        except Exception:
            pass # Default to 0.0, condition will likely be False unless logic below handles it

        if primary_action == 'SELL':
            # Short Position -> SL is Buy above entry
            sl_price = round(((sl_percentage / 100.0) + 1) * avg_price, 1)
            limit_price = round(((5 / 100.0) + 1) * sl_price, 1) # Buy Limit above Trigger
            condition = current_ltp < sl_price if current_ltp > 0 else True # If LTP unknown, assume safe to place SL
        else:
            # Long Position -> SL is Sell below entry
            sl_price = round((1 - (sl_percentage / 100.0)) * avg_price, 1)
            limit_price = round((1 - (5 / 100.0)) * sl_price, 1) # Sell Limit below Trigger
            condition = current_ltp > sl_price if current_ltp > 0 else True # If LTP unknown, assume safe to place SL

        if condition:
            order_response = await self.client.placeorder_async(
                strategy=strategy_tag,
                symbol=symbol,
                action=sl_action,
                exchange=self.exchange,
                price_type="SL",
                product='NRML',
                quantity=qty,
                price=limit_price,
                trigger_price=sl_price
            )
        else:
            # Price already breached, exit immediately
            order_response = await self.async_place_orders(symbol, sl_action, qty, strategy_tag)
            
        if not order_response or order_response.get('status') == 'error':
            print(f"[OpenAlgoClient] SL Order Placement Failed: {order_response}")
            return None

        await asyncio.sleep(0.4)
        order = self.get_order_info_of_order(order_response['orderid'])
        return order

    def subscribe_orderbook(self):
        self.client.subscribe_orderbook(
            api_client=self.client,
            poll_interval=0.3,
            on_data_received=self.on_orderbook_update
        )

    def on_orderbook_update(self, orderbook_data):
        self.orderbook = orderbook_data
