#!/usr/bin/env python3
"""
Image Search System Test & Startup Script
This script validates the complete image search system end-to-end
"""
import os
import sys
import time
import requests
import asyncio
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found. Creating one from template...")
        template_file = Path(".env.template")
        if template_file.exists():
            import shutil
            shutil.copy(".env.template", ".env")
            print("📝 Created .env file from template")
            print("❗ Please edit .env file and add your GEMINI_API_KEY")
            return False
        else:
            print("❌ No .env template found")
            return False
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not configured in .env file")
        print("🔗 Get your API key from: https://aistudio.google.com/")
        return False
    
    print("✅ Environment configuration OK")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn', 
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'google-genai': 'google.genai',
        'pillow': 'PIL', 
        'scikit-learn': 'sklearn',
        'python-multipart': 'multipart',
        'jinja2': 'jinja2'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💻 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("📁 Checking data files...")
    
    csv_file = Path("retail copy.csv")
    if not csv_file.exists():
        print("❌ retail copy.csv not found")
        return False
    
    # Check CSV format
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        required_columns = ['Uniq_Id', 'Product_Name', 'Description', 'Category']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"❌ CSV missing columns: {', '.join(missing_cols)}")
            return False
            
        print(f"✅ CSV file OK ({len(df)} products)")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    return True

def test_services():
    """Test individual services"""
    print("🧪 Testing services...")
    
    try:
        # Test embedding service
        from services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        print("✅ Embedding service initialized")
        
        # Test product database
        from services.product_database import ProductDatabase
        product_db = ProductDatabase()
        product_db.load_products_from_csv()
        print(f"✅ Product database loaded ({len(product_db.products)} products)")
        
        # Test image search service
        from services.image_search_service import ImageSearchService
        image_search_service = ImageSearchService(product_db)
        print("✅ Image search service initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    
    import subprocess
    import sys
    
    # Start server in background
    port = os.environ.get("PORT", "8080")
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", port, "--reload"]
    
    print(f"🌐 Server will start on http://localhost:{port}")
    print("📖 Main app: http://localhost:{port}")
    print("🔍 Image search: http://localhost:{port}/image-search")
    print()
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def run_health_check():
    """Run a quick health check on running server"""
    print("🏥 Running health check...")
    
    port = os.environ.get("PORT", "8080")
    base_url = f"http://localhost:{port}"
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(30):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("❌ Server failed to start within 30 seconds")
        return False
    
    # Test endpoints
    try:
        # Test main page
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ Main page accessible")
        
        # Test image search page
        response = requests.get(f"{base_url}/image-search", timeout=10)
        if response.status_code == 200:
            print("✅ Image search page accessible")
        
        # Test database stats
        response = requests.get(f"{base_url}/products/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Database stats: {stats.get('database_stats', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Main test and startup routine"""
    print("🎯 Image Search System - End-to-End Test & Startup")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        sys.exit(1)
    
    # Step 3: Check data files
    if not check_data_files():
        print("\n❌ Data files check failed. Please ensure retail copy.csv exists.")
        sys.exit(1)
    
    # Step 4: Test services
    if not test_services():
        print("\n❌ Services test failed. Please check your configuration.")
        sys.exit(1)
    
    print("\n✅ All checks passed! System is ready.")
    print("=" * 50)
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Start the server")
    print("2. Run health check (server must be running)")
    print("3. Exit")
    
    choice = input("\nChoose an option (1-3): ").strip()
    
    if choice == "1":
        start_server()
    elif choice == "2":
        run_health_check()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❓ Invalid choice. Exiting.")

if __name__ == "__main__":
    main()