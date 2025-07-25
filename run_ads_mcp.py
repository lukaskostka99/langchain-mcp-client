# Spouštěcí skript pro Google Ads MCP server
import os
import sys
from dotenv import load_dotenv
# Načteme proměnné z .env
load_dotenv()
# Nastavíme port 8001 pro FastMCP před vytvořením instance
os.environ['FASTMCP_PORT'] = '8001'
# Přidáme cestu k MCP Google Ads serveru
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/mcp_servers/mcp-google-ads')))
# Import FastMCP instance
import google_ads_server

if __name__ == "__main__":
    print("Spouštím Google Ads MCP server na http://localhost:8001")
    google_ads_server.mcp.run(transport="sse") 