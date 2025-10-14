#!/bin/zsh

# MIDAS Argo Client Startup Script for macOS
# This script helps launch the complete system

set -e  # Exit on any error

echo "🔬 MIDAS Argo Client Startup"
echo "=============================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV package manager not found. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please create one with your ANL username:"
    echo "   ANL_USERNAME=your_anl_username"
    echo "   ARGO_MODEL=gpt4o"
    exit 1
fi

# Source environment variables (zsh compatible)
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$ANL_USERNAME" ]; then
    echo "❌ ANL_USERNAME not set in .env file"
    exit 1
fi

echo "👤 ANL User: $ANL_USERNAME"
echo "🤖 Default Model: ${ARGO_MODEL:-gpt4o}"

# Function to check if process is running
is_running() {
    pgrep -f "$1" > /dev/null
}

# Function to start MIDAS server
start_midas_server() {
    if is_running "midas_integrated_server.py"; then
        echo "🔧 MIDAS server already running"
        return
    fi
    
    echo "🚀 Starting MIDAS MCP server..."
    uv run python midas_integrated_server.py &
    MIDAS_PID=$!
    sleep 3
    
    if is_running "midas_integrated_server.py"; then
        echo "✅ MIDAS server started (PID: $MIDAS_PID)"
    else
        echo "❌ Failed to start MIDAS server"
        exit 1
    fi
}

# Function to start dashboard server
start_dashboard() {
    if is_running "dashboard_server.py"; then
        echo "🌐 Dashboard server already running"
        return
    fi
    
    echo "🚀 Starting dashboard server..."
    uv run python dashboard_server.py &
    DASHBOARD_PID=$!
    sleep 3
    
    if is_running "dashboard_server.py"; then
        echo "✅ Dashboard server started (PID: $DASHBOARD_PID)"
        echo "🌐 Dashboard available at: http://localhost:8000"
    else
        echo "❌ Failed to start dashboard server"
    fi
}

# Function to start interactive client
start_client() {
    echo "🚀 Starting interactive client..."
    echo "📝 Available commands:"
    echo "   models          - List AI models"
    echo "   model <name>    - Switch model"
    echo "   analyze <file>  - Analyze pattern"
    echo "   tools           - Show MIDAS tools"
    echo "   quit            - Exit"
    echo
    
    uv run python argo_mcp_client.py midas_integrated_server.py
}

# Cleanup function
cleanup() {
    echo
    echo "🛑 Shutting down services..."
    
    if [ ! -z "$MIDAS_PID" ]; then
        kill $MIDAS_PID 2>/dev/null || true
        echo "🔧 MIDAS server stopped"
    fi
    
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null || true
        echo "🌐 Dashboard server stopped"
    fi
    
    # Kill any remaining processes
    pkill -f "midas_integrated_server.py" 2>/dev/null || true
    pkill -f "dashboard_server.py" 2>/dev/null || true
    
    echo "👋 Goodbye!"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Parse command line arguments
case "${1:-interactive}" in
    "server-only")
        echo "🔧 Starting MIDAS server only..."
        start_midas_server
        echo "✅ MIDAS server running. Press Ctrl+C to stop."
        wait
        ;;
        
    "dashboard-only")
        echo "🌐 Starting dashboard only..."
        start_midas_server  # Dashboard needs MIDAS server
        start_dashboard
        echo "✅ Dashboard running. Press Ctrl+C to stop."
        wait
        ;;
        
    "dashboard")
        echo "🌐 Starting full dashboard mode..."
        start_midas_server
        start_dashboard
        echo "✅ All services started. Open http://localhost:8000"
        echo "Press Ctrl+C to stop all services."
        wait
        ;;
        
    "interactive"|"")
        echo "💬 Starting interactive mode..."
        start_midas_server
        start_client
        ;;
        
    "help"|"-h"|"--help")
        echo "Usage: $0 [mode]"
        echo
        echo "Modes:"
        echo "  interactive    - Start servers and interactive client (default)"
        echo "  dashboard      - Start servers and web dashboard"
        echo "  server-only    - Start MIDAS server only"
        echo "  dashboard-only - Start dashboard server only"
        echo "  help          - Show this help"
        echo
        echo "Examples:"
        echo "  $0                    # Interactive mode"
        echo "  $0 dashboard          # Web dashboard mode"  
        echo "  $0 server-only        # Just the MIDAS server"
        ;;
        
    *)
        echo "❌ Unknown mode: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac