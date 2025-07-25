"""
Spouštěcí skript pro Google Analytics MCP server.

Tento skript importuje MCP aplikaci z lokálního zdrojového kódu a spustí ji 
jako standardní webový server pomocí uvicorn na zadaném portu.
"""

import uvicorn
import os
import sys
import json

# Nastavíme cestu k přihlašovacím údajům ještě před importem serveru
# Tím zajistíme, že Google knihovny najdou potřebné údaje
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/lukaskostka/.config/gcloud/application_default_credentials.json'
try:
    # Zkusíme načíst service account JSON z repozitáře pro projekt
    creds_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'googlecloudid.json'))
    with open(creds_json_path) as f:
        creds_data = json.load(f)
        project_id = creds_data.get('project_id')
        if project_id:
            os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
except Exception:
    pass

# Přidáme cestu k našim lokálním MCP serverům, aby je Python našel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/mcp_servers/google-analytics-mcp')))

# Přidáme nastavení portu pro FastMCP
os.environ['FASTMCP_PORT'] = '8000'

# Nyní můžeme bezpečně importovat koordinátora serveru pro GA4
from analytics_mcp.coordinator import mcp as ga4_mcp

# Následující importy jsou nezbytné pro registraci nástrojů.
# Bez nich je objekt 'mcp' nekompletní a nelze ho spustit.
from analytics_mcp.tools.admin import info  # noqa: F401
from analytics_mcp.tools.reporting import realtime  # noqa: F401
from analytics_mcp.tools.reporting import core  # noqa: F401

if __name__ == "__main__":
    print("Spouštím Google Analytics MCP server na http://localhost:8000")
    ga4_mcp.run(transport="sse") 