FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app.py .
COPY weather_server.py .
COPY README.md .

# Expose ports for Streamlit app and MCP server
EXPOSE 8501
EXPOSE 8000

# Create a script to run both the MCP server and Streamlit app
RUN echo '#!/bin/bash\n\
python weather_server.py & \n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/run.sh

RUN chmod +x /app/run.sh

# Run the script when the container launches
CMD ["/app/run.sh"]