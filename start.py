#!/usr/bin/env python3
import uvicorn
import os

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print("🌱 Starting FarmAssist Pro Backend API")
    print(f"🚀 Server running on {host}:{port}")
    print("📊 API Documentation available at:")
    print(f" - Swagger UI: http://{host}:{port}/docs")
    print(f" - ReDoc: http://{host}:{port}/redoc")

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")