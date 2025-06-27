# Example 1: Data Domain-Driven Multi-Agent System with PostgreSQL and BigQuery

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
import asyncio
import json

# Domain Models
@dataclass
class DataSource:
    name: str
    type: str  # 'postgresql', 'bigquery', 'api'
    connection_config: Dict[str, Any]
    schema_info: Optional[Dict[str, Any]] = None

@dataclass
class AgentDomain:
    name: str
    data_sources: List[DataSource]
    capabilities: List[str]
    dependencies: List[str] = None

class AgentRole(Enum):
    DATA_INGESTION = "data_ingestion"
    DATA_PROCESSING = "data_processing" 
    DATA_ANALYSIS = "data_analysis"
    ORCHESTRATION = "orchestration"

# Extensible Agent Factory
class DataDrivenAgentFactory:
    """Factory for creating domain-specific agents with MCP integration"""
    
    def __init__(self, domains: List[AgentDomain]):
        self.domains = {domain.name: domain for domain in domains}
        self.mcp_toolsets = self._initialize_mcp_toolsets()
    
    def _initialize_mcp_toolsets(self) -> Dict[str, MCPToolset]:
        """Initialize MCP toolsets for different data sources"""
        toolsets = {}
        
        for domain in self.domains.values():
            for data_source in domain.data_sources:
                if data_source.type == 'postgresql':
                    toolsets[f"{data_source.name}_postgres"] = MCPToolset(
                        server_params=StdioServerParameters(
                            command="npx",
                            args=["@modelcontextprotocol/server-postgres"],
                            env={
                                "POSTGRES_URL": data_source.connection_config['url'],
                                "POSTGRES_SCHEMA": data_source.connection_config.get('schema', 'public')
                            }
                        )
                    )
                
                elif data_source.type == 'bigquery':
                    toolsets[f"{data_source.name}_bigquery"] = MCPToolset(
                        server_params=StdioServerParameters(
                            command="python",
                            args=["-m", "mcp_toolbox_for_databases.main"],
                            env={
                                "MCP_TOOLBOX_DATABASE_TYPE": "bigquery",
                                "GOOGLE_CLOUD_PROJECT": data_source.connection_config['project_id'],
                                "GOOGLE_APPLICATION_CREDENTIALS": data_source.connection_config['credentials_path']
                            }
                        )
                    )
        
        return toolsets
    
    def create_specialized_agent(self, 
                                role: AgentRole, 
                                domain_name: str,
                                model: str = "gemini-2.0-flash") -> LlmAgent:
        """Create an agent specialized for a specific role and domain"""
        
        domain = self.domains[domain_name]
        tools = []
        
        # Add relevant MCP toolsets based on role and domain
        for data_source in domain.data_sources:
            toolset_key = f"{data_source.name}_{data_source.type}"
            if toolset_key in self.mcp_toolsets:
                tools.append(self.mcp_toolsets[toolset_key])
        
        # Role-specific configuration
        if role == AgentRole.DATA_INGESTION:
            instruction = f"""You are a data ingestion specialist for the {domain_name} domain.
            Your responsibilities:
            1. Extract data from source systems
            2. Validate data quality and completeness
            3. Transform data for downstream processing
            4. Handle errors and data inconsistencies gracefully
            
            Available data sources: {[ds.name for ds in domain.data_sources]}
            """
            
        elif role == AgentRole.DATA_PROCESSING:
            instruction = f"""You are a data processing specialist for the {domain_name} domain.
            Your responsibilities:
            1. Clean and normalize ingested data
            2. Apply business rules and transformations
            3. Aggregate and summarize data as needed
            4. Ensure data consistency across systems
            
            Domain capabilities: {domain.capabilities}
            """
            
        elif role == AgentRole.DATA_ANALYSIS:
            instruction = f"""You are a data analysis specialist for the {domain_name} domain.
            Your responsibilities:
            1. Perform analytical queries and computations
            2. Generate insights and reports
            3. Identify patterns and anomalies
            4. Provide data-driven recommendations
            
            Analysis focus areas: {domain.capabilities}
            """
            
        elif role == AgentRole.ORCHESTRATION:
            instruction = f"""You are an orchestration agent for the {domain_name} domain.
            Your responsibilities:
            1. Coordinate workflows across multiple agents
            2. Route requests to appropriate specialists
            3. Monitor pipeline health and performance
            4. Handle cross-domain dependencies
            
            Managed domains: {list(self.domains.keys())}
            """
        
        return LlmAgent(
            name=f"{domain_name}_{role.value}_agent",
            description=f"{role.value.replace('_', ' ').title()} agent for {domain_name} domain",
            model=model,
            tools=tools,
            instruction=instruction,
            output_key=f"{role.value}_result"
        )

# Example 2: A2A-Enabled Microservice with FastAPI Integration

from fastapi import FastAPI, HTTPException
from google.adk.cli.fast_api import get_fast_api_app
import httpx
import os

class A2AEnabledService:
    """A2A-enabled agent service that can communicate with other agents"""
    
    def __init__(self, 
                 agent_name: str,
                 agent_description: str,
                 mcp_configs: List[Dict[str, Any]],
                 a2a_peers: List[Dict[str, str]] = None):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.mcp_configs = mcp_configs
        self.a2a_peers = a2a_peers or []
        self.app = self._create_app()
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create ADK agent with MCP tools"""
        tools = []
        
        for mcp_config in self.mcp_configs:
            if mcp_config['transport'] == 'stdio':
                params = StdioServerParameters(
                    command=mcp_config['command'],
                    args=mcp_config['args'],
                    env=mcp_config.get('env', {})
                )
                tools.append(MCPToolset(server_params=params))
        
        return LlmAgent(
            name=self.agent_name,
            description=self.agent_description,
            tools=tools,
            model="gemini-2.0-flash"
        )
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with A2A endpoints"""
        
        # Get base ADK FastAPI app
        app = get_fast_api_app(
            agent_dir=os.getcwd(),
            session_db_url=os.getenv('SESSION_DB_URL', 'sqlite:///sessions.db'),
            allow_origins=["*"],
            web=True
        )
        
        # Add A2A discovery endpoint
        @app.get("/.well-known/agent.json")
        async def agent_card():
            return {
                "name": self.agent_name,
                "description": self.agent_description,
                "capabilities": self._get_capabilities(),
                "endpoints": {
                    "run": "/agent/run",
                    "status": "/agent/status",
                    "health": "/health"
                },
                "auth": {
                    "type": "bearer",
                    "required": False
                },
                "version": "0.2"
            }
        
        # Add health check
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "agent": self.agent_name,
                "timestamp": "2025-06-27T00:00:00Z"
            }
        
        # Add A2A communication endpoint
        @app.post("/a2a/communicate")
        async def communicate_with_peer(request: Dict[str, Any]):
            """Communicate with other A2A agents"""
            target_agent = request.get('target_agent')
            message = request.get('message')
            
            # Find peer endpoint
            peer_endpoint = None
            for peer in self.a2a_peers:
                if peer['name'] == target_agent:
                    peer_endpoint = peer['endpoint']
                    break
            
            if not peer_endpoint:
                raise HTTPException(status_code=404, detail=f"Agent {target_agent} not found")
            
            # Send A2A request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{peer_endpoint}/agent/run",
                    json={
                        "jsonrpc": "2.0",
                        "method": "run",
                        "params": {
                            "message": message,
                            "sender": self.agent_name
                        },
                        "id": 1
                    }
                )
            
            return response.json()
        
        return app
    
    def _get_capabilities(self) -> List[str]:
        """Extract capabilities from MCP tools"""
        capabilities = []
        for mcp_config in self.mcp_configs:
            capabilities.extend(mcp_config.get('capabilities', []))
        return capabilities

# Example 3: Multi-Protocol Integration Pattern

class UnifiedMultiAgentOrchestrator:
    """Orchestrator that can communicate via both A2A and direct MCP"""
    
    def __init__(self):
        self.a2a_agents = {}  # Remote A2A agents
        self.local_agents = {}  # Local ADK agents
        self.mcp_servers = {}  # Direct MCP servers
    
    def register_a2a_agent(self, name: str, endpoint: str):
        """Register a remote A2A agent"""
        self.a2a_agents[name] = {
            'endpoint': endpoint,
            'capabilities': self._discover_a2a_capabilities(endpoint)
        }
    
    def register_local_agent(self, name: str, agent: LlmAgent):
        """Register a local ADK agent"""
        self.local_agents[name] = agent
    
    def register_mcp_server(self, name: str, toolset: MCPToolset):
        """Register a direct MCP server connection"""
        self.mcp_servers[name] = toolset
    
    async def execute_distributed_workflow(self, workflow_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow across multiple agents and protocols"""
        
        results = {}
        
        for step in workflow_spec['steps']:
            step_name = step['name']
            agent_type = step['agent_type']  # 'a2a', 'local', 'mcp'
            agent_name = step['agent_name']
            task = step['task']
            
            if agent_type == 'a2a':
                # Execute via A2A protocol
                result = await self._execute_a2a_task(agent_name, task)
            
            elif agent_type == 'local':
                # Execute via local ADK agent
                result = await self._execute_local_task(agent_name, task)
            
            elif agent_type == 'mcp':
                # Execute via direct MCP
                result = await self._execute_mcp_task(agent_name, task)
            
            results[step_name] = result
            
            # Pass results to next step if needed
            if 'output_mapping' in step:
                self._apply_output_mapping(step['output_mapping'], result, workflow_spec)
        
        return results
    
    async def _execute_a2a_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task via A2A protocol"""
        agent_info = self.a2a_agents[agent_name]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{agent_info['endpoint']}/agent/run",
                json={
                    "jsonrpc": "2.0",
                    "method": "run",
                    "params": task,
                    "id": 1
                }
            )
        
        return response.json()
    
    async def _execute_local_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task via local ADK agent"""
        agent = self.local_agents[agent_name]
        
        # Create invocation context
        from google.adk.core.invocation_context import InvocationContext
        context = InvocationContext()
        
        # Execute agent
        result = await agent.invoke(context, task['message'])
        return {"result": result, "status": "success"}
    
    async def _execute_mcp_task(self, server_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task via direct MCP server"""
        mcp_toolset = self.mcp_servers[server_name]
        
        # This would require implementing direct MCP tool calls
        # For now, we'll create a temporary agent to use the toolset
        temp_agent = LlmAgent(
            name="temp_mcp_agent",
            tools=[mcp_toolset],
            model="gemini-2.0-flash"
        )
        
        from google.adk.core.invocation_context import InvocationContext
        context = InvocationContext()
        result = await temp_agent.invoke(context, task['message'])
        
        return {"result": result, "status": "success"}
    
    def _discover_a2a_capabilities(self, endpoint: str) -> List[str]:
        """Discover capabilities from A2A agent card"""
        import requests
        try:
            response = requests.get(f"{endpoint}/.well-known/agent.json")
            agent_card = response.json()
            return agent_card.get('capabilities', [])
        except Exception as e:
            print(f"Failed to discover capabilities for {endpoint}: {e}")
            return []
    
    def _apply_output_mapping(self, mapping: Dict[str, str], result: Dict[str, Any], workflow_spec: Dict[str, Any]):
        """Apply output mapping to pass results between workflow steps"""
        for source_key, target_key in mapping.items():
            if source_key in result:
                # Store result for use in subsequent steps
                if 'context' not in workflow_spec:
                    workflow_spec['context'] = {}
                workflow_spec['context'][target_key] = result[source_key]

# Example 4: Production-Ready Multi-Agent System for Data Analytics

class DataAnalyticsMultiAgentSystem:
    """Production-ready multi-agent system for data analytics workflows"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.agent_factory = DataDrivenAgentFactory(self._parse_domains())
        self.orchestrator = UnifiedMultiAgentOrchestrator()
        self._setup_agents()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _parse_domains(self) -> List[AgentDomain]:
        """Parse domain configurations"""
        domains = []
        
        for domain_config in self.config['domains']:
            data_sources = []
            for ds_config in domain_config['data_sources']:
                data_sources.append(DataSource(
                    name=ds_config['name'],
                    type=ds_config['type'],
                    connection_config=ds_config['connection']
                ))
            
            domains.append(AgentDomain(
                name=domain_config['name'],
                data_sources=data_sources,
                capabilities=domain_config['capabilities'],
                dependencies=domain_config.get('dependencies', [])
            ))
        
        return domains
    
    def _setup_agents(self):
        """Initialize all agents based on configuration"""
        
        # Create local specialized agents
        for domain_name in self.config['domains']:
            domain_name = domain_name['name']
            
            # Create agents for each role
            ingestion_agent = self.agent_factory.create_specialized_agent(
                AgentRole.DATA_INGESTION, domain_name
            )
            processing_agent = self.agent_factory.create_specialized_agent(
                AgentRole.DATA_PROCESSING, domain_name
            )
            analysis_agent = self.agent_factory.create_specialized_agent(
                AgentRole.DATA_ANALYSIS, domain_name
            )
            
            # Register with orchestrator
            self.orchestrator.register_local_agent(f"{domain_name}_ingestion", ingestion_agent)
            self.orchestrator.register_local_agent(f"{domain_name}_processing", processing_agent)
            self.orchestrator.register_local_agent(f"{domain_name}_analysis", analysis_agent)
        
        # Register remote A2A agents if configured
        if 'a2a_agents' in self.config:
            for a2a_config in self.config['a2a_agents']:
                self.orchestrator.register_a2a_agent(
                    a2a_config['name'], 
                    a2a_config['endpoint']
                )
    
    async def execute_analytics_pipeline(self, pipeline_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a predefined analytics pipeline"""
        
        pipeline_config = self.config['pipelines'][pipeline_name]
        
        # Substitute parameters in workflow
        workflow_spec = self._substitute_parameters(pipeline_config['workflow'], parameters)
        
        # Execute distributed workflow
        results = await self.orchestrator.execute_distributed_workflow(workflow_spec)
        
        # Post-process results if needed
        if 'post_processing' in pipeline_config:
            results = await self._post_process_results(results, pipeline_config['post_processing'])
        
        return results
    
    def _substitute_parameters(self, workflow: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute parameters in workflow specification"""
        import copy
        workflow_copy = copy.deepcopy(workflow)
        
        # Simple parameter substitution
        workflow_str = json.dumps(workflow_copy)
        for param_name, param_value in parameters.items():
            workflow_str = workflow_str.replace(f"${{{param_name}}}", str(param_value))
        
        return json.loads(workflow_str)
    
    async def _post_process_results(self, results: Dict[str, Any], post_processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply post-processing to pipeline results"""
        
        if post_processing_config.get('aggregate'):
            # Aggregate results from multiple steps
            aggregated = {}
            for step_name, result in results.items():
                if 'result' in result:
                    aggregated[step_name] = result['result']
            results['aggregated'] = aggregated
        
        if post_processing_config.get('format') == 'report':
            # Generate a formatted report
            report_agent = self.agent_factory.create_specialized_agent(
                AgentRole.DATA_ANALYSIS, 
                "reporting"
            )
            
            from google.adk.core.invocation_context import InvocationContext
            context = InvocationContext()
            
            report_prompt = f"""
            Generate a comprehensive analytics report based on the following results:
            {json.dumps(results, indent=2)}
            
            Include:
            1. Executive summary
            2. Key findings
            3. Data quality assessment
            4. Recommendations
            5. Next steps
            """
            
            report = await report_agent.invoke(context, report_prompt)
            results['report'] = report
        
        return results

# Example 5: Kubernetes Deployment Configuration

class KubernetesMultiAgentDeployer:
    """Deploy multi-agent system to Kubernetes with proper service discovery"""
    
    def __init__(self, namespace: str = "ai-agents"):
        self.namespace = namespace
        self.agent_services = []
    
    def generate_deployment_manifests(self, agents_config: Dict[str, Any]) -> Dict[str, str]:
        """Generate Kubernetes manifests for multi-agent deployment"""
        
        manifests = {}
        
        # Generate namespace
        manifests['namespace.yaml'] = self._generate_namespace()
        
        # Generate ConfigMap for shared configuration
        manifests['configmap.yaml'] = self._generate_configmap(agents_config)
        
        # Generate Secret for credentials
        manifests['secret.yaml'] = self._generate_secret(agents_config)
        
        # Generate deployments and services for each agent
        for agent_name, agent_config in agents_config['agents'].items():
            
            # Deployment
            manifests[f'deployment-{agent_name}.yaml'] = self._generate_deployment(
                agent_name, agent_config
            )
            
            # Service
            manifests[f'service-{agent_name}.yaml'] = self._generate_service(
                agent_name, agent_config
            )
            
            # If agent exposes A2A endpoints, create Ingress
            if agent_config.get('expose_a2a', False):
                manifests[f'ingress-{agent_name}.yaml'] = self._generate_ingress(
                    agent_name, agent_config
                )
        
        # Generate orchestrator deployment
        manifests['deployment-orchestrator.yaml'] = self._generate_orchestrator_deployment(agents_config)
        manifests['service-orchestrator.yaml'] = self._generate_orchestrator_service()
        
        return manifests
    
    def _generate_namespace(self) -> str:
        return f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.namespace}
  labels:
    name: {self.namespace}
"""
    
    def _generate_configmap(self, config: Dict[str, Any]) -> str:
        return f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: {self.namespace}
data:
  agents-config.json: |
{json.dumps(config, indent=4)}
"""
    
    def _generate_secret(self, config: Dict[str, Any]) -> str:
        import base64
        
        secrets_data = {}
        
        # Encode database credentials
        if 'database_credentials' in config:
            for db_name, creds in config['database_credentials'].items():
                secrets_data[f"{db_name}-url"] = base64.b64encode(
                    creds['url'].encode()
                ).decode()
        
        # Encode API keys
        if 'api_keys' in config:
            for service, key in config['api_keys'].items():
                secrets_data[f"{service}-api-key"] = base64.b64encode(
                    key.encode()
                ).decode()
        
        secrets_yaml = """
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: {namespace}
type: Opaque
data:
""".format(namespace=self.namespace)
        
        for key, value in secrets_data.items():
            secrets_yaml += f"  {key}: {value}\n"
        
        return secrets_yaml
    
    def _generate_deployment(self, agent_name: str, agent_config: Dict[str, Any]) -> str:
        
        # Environment variables for MCP servers
        env_vars = []
        if 'mcp_servers' in agent_config:
            for mcp_name, mcp_config in agent_config['mcp_servers'].items():
                if mcp_config['type'] == 'postgresql':
                    env_vars.append(f"""
        - name: {mcp_name.upper()}_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: {mcp_name}-url""")
                
                elif mcp_config['type'] == 'bigquery':
                    env_vars.append(f"""
        - name: GOOGLE_CLOUD_PROJECT
          value: "{mcp_config['project_id']}"
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/etc/credentials/service-account.json" """)
        
        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {agent_name}-agent
  namespace: {self.namespace}
spec:
  replicas: {agent_config.get('replicas', 2)}
  selector:
    matchLabels:
      app: {agent_name}-agent
  template:
    metadata:
      labels:
        app: {agent_name}-agent
    spec:
      containers:
      - name: {agent_name}-agent
        image: {agent_config.get('image', 'gcr.io/project/adk-agent:latest')}
        ports:
        - containerPort: 8080
        env:
        - name: AGENT_NAME
          value: "{agent_name}"
        - name: MODEL_NAME
          value: "{agent_config.get('model', 'gemini-2.0-flash')}"
        - name: CONFIG_PATH
          value: "/etc/config/agents-config.json"{''.join(env_vars)}
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: secrets-volume
          mountPath: /etc/secrets
        resources:
          requests:
            memory: "{agent_config.get('memory_request', '512Mi')}"
            cpu: "{agent_config.get('cpu_request', '250m')}"
          limits:
            memory: "{agent_config.get('memory_limit', '1Gi')}"
            cpu: "{agent_config.get('cpu_limit', '500m')}"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: agent-config
      - name: secrets-volume
        secret:
          secretName: agent-secrets
"""
    
    def _generate_service(self, agent_name: str, agent_config: Dict[str, Any]) -> str:
        return f"""
apiVersion: v1
kind: Service
metadata:
  name: {agent_name}-agent-service
  namespace: {self.namespace}
spec:
  selector:
    app: {agent_name}-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
"""
    
    def _generate_ingress(self, agent_name: str, agent_config: Dict[str, Any]) -> str:
        domain = agent_config.get('domain', 'agents.example.com')
        return f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {agent_name}-agent-ingress
  namespace: {self.namespace}
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  tls:
  - hosts:
    - {agent_name}.{domain}
    secretName: {agent_name}-agent-tls
  rules:
  - host: {agent_name}.{domain}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {agent_name}-agent-service
            port:
              number: 80
"""
    
    def _generate_orchestrator_deployment(self, config: Dict[str, Any]) -> str:
        
        # Build A2A discovery endpoints
        a2a_endpoints = []
        for agent_name in config['agents'].keys():
            a2a_endpoints.append(f"http://{agent_name}-agent-service.{self.namespace}.svc.cluster.local")
        
        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
  namespace: {self.namespace}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: gcr.io/project/orchestrator:latest
        ports:
        - containerPort: 8080
        env:
        - name: A2A_DISCOVERY_ENDPOINTS
          value: "{','.join(a2a_endpoints)}"
        - name: CONFIG_PATH
          value: "/etc/config/agents-config.json"
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config-volume
        configMap:
          name: agent-config
"""
    
    def _generate_orchestrator_service(self) -> str:
        return f"""
apiVersion: v1
kind: Service
metadata:
  name: orchestrator-service
  namespace: {self.namespace}
spec:
  selector:
    app: orchestrator
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
"""

# Example usage and configuration

if __name__ == "__main__":
    
    # Example configuration for a data analytics system
    system_config = {
        "domains": [
            {
                "name": "sales",
                "data_sources": [
                    {
                        "name": "sales_db",
                        "type": "postgresql",
                        "connection": {
                            "url": "postgresql://user:pass@sales-db:5432/sales",
                            "schema": "public"
                        }
                    },
                    {
                        "name": "sales_analytics",
                        "type": "bigquery",
                        "connection": {
                            "project_id": "my-project",
                            "credentials_path": "/etc/credentials/bigquery.json"
                        }
                    }
                ],
                "capabilities": ["revenue_analysis", "customer_segmentation", "sales_forecasting"]
            },
            {
                "name": "inventory",
                "data_sources": [
                    {
                        "name": "inventory_db",
                        "type": "postgresql", 
                        "connection": {
                            "url": "postgresql://user:pass@inventory-db:5432/inventory",
                            "schema": "public"
                        }
                    }
                ],
                "capabilities": ["stock_analysis", "demand_forecasting", "supply_chain_optimization"]
            }
        ],
        "pipelines": {
            "monthly_sales_analysis": {
                "workflow": {
                    "steps": [
                        {
                            "name": "extract_sales_data",
                            "agent_type": "local",
                            "agent_name": "sales_ingestion",
                            "task": {
                                "message": "Extract sales data for month ${month} year ${year}"
                            },
                            "output_mapping": {
                                "result": "sales_data"
                            }
                        },
                        {
                            "name": "process_sales_data",
                            "agent_type": "local", 
                            "agent_name": "sales_processing",
                            "task": {
                                "message": "Clean and process sales data: ${sales_data}"
                            },
                            "output_mapping": {
                                "result": "processed_sales"
                            }
                        },
                        {
                            "name": "analyze_sales_trends",
                            "agent_type": "local",
                            "agent_name": "sales_analysis", 
                            "task": {
                                "message": "Analyze sales trends and patterns: ${processed_sales}"
                            }
                        }
                    ]
                },
                "post_processing": {
                    "aggregate": True,
                    "format": "report"
                }
            }
        },
        "agents": {
            "sales-agent": {
                "image": "gcr.io/my-project/sales-agent:latest",
                "model": "gemini-2.0-flash",
                "replicas": 2,
                "expose_a2a": True,
                "domain": "agents.mycompany.com",
                "mcp_servers": {
                    "sales_db": {
                        "type": "postgresql"
                    },
                    "sales_analytics": {
                        "type": "bigquery",
                        "project_id": "my-project"
                    }
                }
            },
            "inventory-agent": {
                "image": "gcr.io/my-project/inventory-agent:latest", 
                "model": "gemini-2.0-flash",
                "replicas": 2,
                "expose_a2a": True,
                "domain": "agents.mycompany.com",
                "mcp_servers": {
                    "inventory_db": {
                        "type": "postgresql"
                    }
                }
            }
        }
    }
    
    # Initialize the system
    analytics_system = DataAnalyticsMultiAgentSystem("config.json")
    
    # Execute a pipeline
    async def run_analysis():
        results = await analytics_system.execute_analytics_pipeline(
            "monthly_sales_analysis",
            {"month": "06", "year": "2025"}
        )
        print("Analysis Results:", json.dumps(results, indent=2))
    
    # For deployment
    deployer = KubernetesMultiAgentDeployer("ai-agents")
    manifests = deployer.generate_deployment_manifests(system_config)
    
    # Save manifests to files
    for filename, content in manifests.items():
        with open(f"k8s/{filename}", 'w') as f:
            f.write(content)