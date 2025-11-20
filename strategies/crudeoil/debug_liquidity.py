from openalgo_client import OpenAlgoClientWrapper
from config import default_config as cfg
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DEBUG_LIQUIDITY")

def debug_depth():
    client = OpenAlgoClientWrapper(
        api_key=cfg.OPENALGO_API_KEY,
        host=cfg.OPENALGO_HOST,
        ws_url=cfg.OPENALGO_WS,
        on_ltp_callback=lambda x: None
    )
    
    symbol = "CRUDEOILM16DEC255300CE"
    
    logger.info(f"Fetching depth for {symbol}...")
    try:
        # Test 1: Standard get_market_depth
        depth = client.get_market_depth(symbol)
        logger.info(f"Raw Depth Response: {depth}")
        
        if depth:
            bids = depth.get('bids', [])
            asks = depth.get('asks', [])
            logger.info(f"Bids: {len(bids)}, Asks: {len(asks)}")
            if bids: logger.info(f"Best Bid: {bids[0]}")
            if asks: logger.info(f"Best Ask: {asks[0]}")
        else:
            logger.warning("Depth is None or Empty")

        # Test 2: Check get_ltp
        logger.info("Fetching LTP...")
        try:
            ltp_data = client.client.get_ltp(exchange="MCX", symbol=symbol)
            logger.info(f"LTP Data: {ltp_data}")
        except Exception as e:
            logger.error(f"LTP Fetch Error: {e}")

        # Test 3: Check if symbol is valid in quotes
        logger.info("Fetching Quote...")
        # quote = client.get_quote(symbol) # Wrapper doesn't have it, try raw
        quote = client.client.quotes(symbol=symbol, exchange="MCX")
        logger.info(f"Quote: {quote}")

    except Exception as e:
        logger.error(f"Exception: {e}")

if __name__ == "__main__":
    debug_depth()
