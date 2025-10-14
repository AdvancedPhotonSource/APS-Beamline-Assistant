#!/bin/zsh

# Debug script to troubleshoot MIDAS startup issues

echo "🔍 MIDAS Debug - Troubleshooting Startup Issues"
echo "==============================================="

# Test Python and UV
echo "1. Testing Python environment..."
echo "Python version: $(python3 --version 2>&1)"
echo "UV version: $(uv --version 2>&1)"
echo "Current directory: $(pwd)"

# Test dependencies
echo -e "\n2. Testing dependencies..."
echo "Testing MCP import:"
uv run python3 -c "import mcp; print('✅ MCP OK')" 2>&1

echo "Testing httpx import:"
uv run python3 -c "import httpx; print('✅ httpx OK')" 2>&1

echo "Testing asyncio:"
uv run python3 -c "import asyncio; print('✅ asyncio OK')" 2>&1

# Test server script
echo -e "\n3. Testing server script..."
if [ -f "simple_midas_server.py" ]; then
    echo "✅ simple_midas_server.py found"
    echo "Testing syntax:"
    python3 -m py_compile simple_midas_server.py 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Syntax OK"
    else
        echo "❌ Syntax error in simple_midas_server.py"
    fi
else
    echo "❌ simple_midas_server.py not found"
fi

# Test client script
echo -e "\n4. Testing client script..."
if [ -f "argo_mcp_client.py" ]; then
    echo "✅ argo_mcp_client.py found"
    echo "Testing syntax:"
    python3 -m py_compile argo_mcp_client.py 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Syntax OK"
    else
        echo "❌ Syntax error in argo_mcp_client.py"
    fi
else
    echo "❌ argo_mcp_client.py not found"
fi

# Test environment file
echo -e "\n5. Testing environment..."
if [ -f ".env" ]; then
    echo "✅ .env file found"
    if grep -q "ANL_USERNAME=" .env && ! grep -q "ANL_USERNAME=your_anl_username" .env; then
        echo "✅ ANL_USERNAME configured"
    else
        echo "❌ ANL_USERNAME not properly configured"
    fi
else
    echo "❌ .env file not found"
fi

# Test network connectivity
echo -e "\n6. Testing network connectivity..."
if ping -c 1 apps.inside.anl.gov &> /dev/null; then
    echo "✅ Can reach apps.inside.anl.gov"
else
    echo "❌ Cannot reach apps.inside.anl.gov - check VPN"
fi

# Test running the simple server directly
echo -e "\n7. Testing server startup..."
echo "Attempting to start simple server for 5 seconds..."

# Start server in background and test
(timeout 5s uv run python3 simple_midas_server.py &) 
SERVER_PID=$!
sleep 2

if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ Server started successfully"
    kill $SERVER_PID 2>/dev/null
else
    echo "❌ Server failed to start"
    echo "Trying to get error output:"
    uv run python3 simple_midas_server.py &
    sleep 2
    kill %1 2>/dev/null
fi

echo -e "\n8. Manual test commands:"
echo "Try these commands manually:"
echo ""
echo "# Test server directly:"
echo "uv run python3 simple_midas_server.py"
echo ""
echo "# Test client directly:"  
echo "uv run python3 argo_mcp_client.py simple_midas_server.py"
echo ""
echo "# Check for running processes:"
echo "ps aux | grep -E '(midas|python)'"
echo ""

echo "✅ Debug complete. Check output above for issues."