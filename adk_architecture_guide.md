# Google ADK Multi-Agent & MCP Server Architecture

## Architecture Overview

The Google ADK architecture supports three main communication patterns:

1. **Intra-Agent Communication**: Agents within the same ADK application using hierarchical structures
2. **Inter-Agent Communication**: Distributed agents across services using A2A (Agent-to-Agent) protocol  
3. **Agent-to-Tool Communication**: Agents connecting to external capabilities via MCP servers

## Core Components

### 1. Agent Development Kit (ADK) Core
- **LlmAgent**: Primary agent class with LLM capabilities
- **BaseAgent**: Custom non-LLM agents for specialized logic
- **SequentialAgent**: Pipeline orchestration
- **LoopAgent**: Iterative workflows
- **MCPToolset**: Integration layer for MCP servers

### 2. Agent-to-Agent (A2A) Protocol
- **Client Agent**: Initiates requests to other agents
- **Remote Agent**: Receives and processes requests from other agents
- **Agent Card**: JSON metadata describing agent capabilities
- **A2A Inspector**: Development tool for debugging agent interactions

### 3. Model Context Protocol (MCP) Integration
- **MCP Client**: ADK agents acting as MCP clients
- **MCP Server**: External services exposing tools/data
- **MCPToolset**: ADK's mechanism for integrating MCP tools
- **Transport Layers**: STDIO, SSE, HTTP for different deployment scenarios

## Deployment Patterns

### Pattern 1: Monolithic Multi-Agent System

<pre>
┌─────────────────────────────────────────┐
│           ADK Application               │
│  ┌─────────────┐  ┌─────────────┐       │
│  │ Coordinator │  │ Specialist  │       │
│  │   Agent     │  │   Agents    │       │
│  └─────────────┘  └─────────────┘       │
│         │               │               │
│         └───────────────┘               │
│                 │                       │
│         ┌───────▼───────┐               │
│         │  MCP Toolset  │               │
│         └───────────────┘               │
└─────────────────────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │    External MCP Servers │
    │  ┌─────┐ ┌─────┐ ┌─────┐│
    │  │ DB  │ │ API │ │File ││
    │  │ MCP │ │ MCP │ │ MCP ││
    │  └─────┘ └─────┘ └─────┘│
    └─────────────────────────┘
</pre>

### Pattern 2: Distributed Multi-Agent Architecture

<pre>
┌─────────────────┐    A2A Protocol    ┌─────────────────┐
│   Client Agent  │◄─────────────────►│  Remote Agent    │
│   (ADK/FastAPI) │     HTTP/JSON-RPC  │   (ADK/FastAPI) │
└─────────────────┘                    └─────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────┐                    ┌─────────────────┐
│   MCP Servers   │                    │   MCP Servers   │
│ ┌─────┐ ┌─────┐ │                    │ ┌─────┐ ┌─────┐ │
│ │ DB  │ │ API │ │                    │ │Files│ │Auth │ │
│ └─────┘ └─────┘ │                    │ └─────┘ └─────┘ │
└─────────────────┘                    └─────────────────┘
</pre>

### Pattern 3: Microservices with Central Orchestration

<pre>
┌─────────────────────────────────────────────────────────┐
│                Orchestrator Agent                       │
│                   (ADK Core)                            │
└────────────────────┬────────────────────────────────────┘
                     │ A2A Protocol
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Flight Agent│ │ Hotel Agent │ │ Weather Agt │
│ (Cloud Run) │ │ (Cloud Run) │ │ (Cloud Run) │
└─────────────┘ └─────────────┘ └─────────────┘
        │            │            │
        ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Flight MCP   │ │Booking MCP  │ │Weather MCP  │
│ Server      │ │ Server      │ │ Server      │
└─────────────┘ └─────────────┘ └─────────────┘
</pre>

## Data Flow Architecture

### 1. Agent Hierarchy Communication
```python
# Intra-agent communication via state sharing
agent_a = LlmAgent(
    name="DataProcessor", 
    output_key="processed_data"
)
agent_b = LlmAgent(
    name="DataAnalyzer", 
    instruction="Analyze data from state key 'processed_data'"
)

coordinator = SequentialAgent(
    name="DataPipeline",
    sub_agents=[agent_a, agent_b]
)
```

### 2. A2A Inter-Agent Communication
```python
# Agent Card Discovery & Communication
{
  "name": "DatabaseAgent",
  "description": "Handles all database operations",
  "capabilities": ["query", "update", "schema_analysis"],
  "endpoints": {
    "run": "/agent/run",
    "status": "/agent/status"
  },
  "auth": {
    "type": "bearer",
    "required": true
  }
}
```

### 3. MCP Integration Flow
```python
# MCPToolset integration
database_tools = MCPToolset(
    server_params=StdioServerParameters(
        command="npx",
        args=["@modelcontextprotocol/server-postgres"],
        env={"DATABASE_URL": "postgresql://..."}
    )
)

agent = LlmAgent(
    name="DatabaseAgent",
    tools=[database_tools],
    model="gemini-2.0-flash"
)
```

## Implementation Approaches

### Approach 1: Domain-Driven Multi-Agent Design

**Data Domain Models:**
```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class AgentType(Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    GATEWAY = "gateway"

@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: dict
    output_schema: dict

@dataclass
class AgentDomain:
    domain: str
    agents: List['DomainAgent']
    mcp_servers: List['MCPServerConfig']
    
@dataclass
class MCPServerConfig:
    name: str
    transport: str  # "stdio", "sse", "http"
    connection_params: dict
    capabilities: List[str]
```

**Extensible Agent Factory:**
```python
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

class AgentFactory:
    def __init__(self, domain_config: AgentDomain):
        self.domain_config = domain_config
        self.mcp_toolsets = self._initialize_mcp_toolsets()
    
    def _initialize_mcp_toolsets(self) -> dict:
        toolsets = {}
        for mcp_config in self.domain_config.mcp_servers:
            if mcp_config.transport == "stdio":
                params = StdioServerParameters(**mcp_config.connection_params)
            elif mcp_config.transport == "sse":
                params = SSEServerParameters(**mcp_config.connection_params)
            
            toolsets[mcp_config.name] = MCPToolset(server_params=params)
        return toolsets
    
    def create_agent(self, agent_config: dict) -> LlmAgent:
        tools = []
        
        # Add MCP toolsets based on agent requirements
        for mcp_name in agent_config.get('mcp_tools', []):
            if mcp_name in self.mcp_toolsets:
                tools.append(self.mcp_toolsets[mcp_name])
        
        # Add custom tools
        for tool_name in agent_config.get('custom_tools', []):
            tools.append(self._create_custom_tool(tool_name))
        
        return LlmAgent(
            name=agent_config['name'],
            description=agent_config['description'],
            model=agent_config.get('model', 'gemini-2.0-flash'),
            tools=tools,
            instruction=agent_config.get('instruction', '')
        )
```

### Approach 2: A2A-Enabled Microservice Architecture

**FastAPI Agent Wrapper:**
```python
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.agents import LlmAgent
import os

class A2AAgentService:
    def __init__(self, agent_config: dict, mcp_configs: List[dict]):
        self.agent_config = agent_config
        self.mcp_configs = mcp_configs
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        # Create ADK agent with MCP integration
        agent = self._create_agent_with_mcp()
        
        # Get FastAPI app with ADK integration
        app = get_fast_api_app(
            agent_dir=os.getcwd(),
            session_db_url="postgresql://localhost/agent_sessions",
            allow_origins=["*"],
            web=True
        )
        
        # Add A2A endpoints
        self._add_a2a_endpoints(app)
        return app
    
    def _create_agent_with_mcp(self) -> LlmAgent:
        tools = []
        for mcp_config in self.mcp_configs:
            toolset = MCPToolset(
                server_params=self._create_mcp_params(mcp_config)
            )
            tools.append(toolset)
        
        return LlmAgent(
            name=self.agent_config['name'],
            description=self.agent_config['description'],
            tools=tools,
            model=self.agent_config.get('model', 'gemini-2.0-flash')
        )
    
    def _add_a2a_endpoints(self, app: FastAPI):
        @app.get("/.well-known/agent.json")
        async def agent_card():
            return {
                "name": self.agent_config['name'],
                "description": self.agent_config['description'],
                "capabilities": self.agent_config.get('capabilities', []),
                "endpoints": {
                    "run": "/agent/run",
                    "status": "/agent/status"
                }
            }
```

### Approach 3: Multi-Protocol Integration Pattern

**Unified Communication Layer:**
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class CommunicationProtocol(ABC):
    @abstractmethod
    async def send_request(self, target: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        pass

class A2AProtocol(CommunicationProtocol):
    async def send_request(self, target: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # A2A JSON-RPC implementation
        return await self._send_a2a_request(target, payload)

class MCPProtocol(CommunicationProtocol):
    async def send_request(self, target: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # MCP client implementation
        return await self._send_mcp_request(target, payload)

class UnifiedAgent:
    def __init__(self, name: str, protocols: List[CommunicationProtocol]):
        self.name = name
        self.protocols = {type(p).__name__: p for p in protocols}
        self.capabilities = self._discover_capabilities()
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Route task based on required protocol
        protocol_type = self._determine_protocol(task)
        protocol = self.protocols[protocol_type]
        return await protocol.send_request(task['target'], task['payload'])
```

## Deployment on Google Cloud

### Cloud Run Deployment
```yaml
# cloud-run-agent.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: adk-agent-service
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
    spec:
      containers:
      - image: gcr.io/project/adk-agent:latest
        env:
        - name: MODEL_NAME
          value: "gemini-2.0-flash"
        - name: MCP_SERVERS
          value: "database,files,api"
        ports:
        - containerPort: 8080
```

### Kubernetes Multi-Agent Deployment
```yaml
# k8s-multi-agent.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    spec:
      containers:
      - name: orchestrator
        image: gcr.io/project/orchestrator-agent:latest
        env:
        - name: A2A_DISCOVERY_ENDPOINTS
          value: "http://flight-agent,http://hotel-agent,http://weather-agent"
---
apiVersion: v1
kind: Service
metadata:
  name: agent-orchestrator
spec:
  selector:
    app: agent-orchestrator
  ports:
  - port: 80
    targetPort: 8080
```

## Benefits of This Architecture

1. **Scalability**: Each agent can be scaled independently
2. **Resilience**: Failure of one agent doesn't affect others
3. **Flexibility**: Mix and match agents from different frameworks
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add new agents and MCP servers
6. **Interoperability**: A2A protocol enables cross-platform communication

## Security Considerations

- **Authentication**: A2A supports OpenAPI authentication schemes
- **Authorization**: Agent-level permissions and capabilities
- **Network Security**: TLS encryption for all communications
- **Data Privacy**: Agents don't share internal state or memory
- **Audit Logging**: Full traceability of agent interactions

This architecture provides a robust foundation for building enterprise-grade multi-agent systems that can scale horizontally while maintaining loose coupling and high cohesion.