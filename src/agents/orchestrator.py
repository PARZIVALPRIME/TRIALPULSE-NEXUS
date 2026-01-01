"""
Agent Orchestrator for TRIALPULSE NEXUS 10X
Phase 5.2: LangGraph Agent Framework - FIXED

Defines the LangGraph workflow that orchestrates all agents.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import (
    AgentState, AgentType, TaskStatus, TaskPriority,
    create_initial_state, Recommendation, Hypothesis, Evidence, Forecast, Communication
)
from src.agents.base_agent import BaseAgent
from src.agents.llm_wrapper import LLMWrapper, get_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# SHARED LLM INSTANCE (Efficiency fix)
# ============================================================

_shared_llm: Optional[LLMWrapper] = None


def get_shared_llm() -> LLMWrapper:
    """Get a shared LLM instance for all agents"""
    global _shared_llm
    if _shared_llm is None:
        _shared_llm = get_llm()
    return _shared_llm


# ============================================================
# AGENT IMPLEMENTATIONS
# ============================================================

class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent - Orchestrates other agents
    
    Responsibilities:
    - Analyze user query
    - Decompose into subtasks
    - Route to appropriate agents
    - Synthesize final response
    """
    
    def __init__(self, **kwargs):
        # Use shared LLM
        kwargs['llm'] = get_shared_llm()
        super().__init__(AgentType.SUPERVISOR, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are the Supervisor Agent for a clinical trial management system.

Your responsibilities:
1. Analyze the user's query to understand their intent
2. Decompose complex requests into subtasks
3. Determine which specialist agents should handle each part
4. Synthesize information from other agents into a coherent response

Available specialist agents:
- DIAGNOSTIC: Investigates issues, forms hypotheses, identifies root causes
- FORECASTER: Predicts timelines, estimates effort, projects outcomes
- RESOLVER: Creates action plans, prioritizes tasks, recommends solutions
- EXECUTOR: Validates and executes approved actions
- COMMUNICATOR: Drafts messages, notifications, and reports

You coordinate but never make final decisions yourself.
Always route to specialists for analysis."""
    
    @property
    def allowed_tools(self) -> List[str]:
        return ["get_study_summary", "get_high_priority_patients", "get_overall_summary"]
    
    def _determine_agents_needed(self, query: str, state: AgentState) -> List[AgentType]:
        """Determine which agents are needed based on the query"""
        query_lower = query.lower()
        agents = []
        
        # Keywords to agent mapping
        if any(w in query_lower for w in ["why", "cause", "reason", "investigate", "analyze", "issue", "problem"]):
            agents.append(AgentType.DIAGNOSTIC)
        
        if any(w in query_lower for w in ["when", "timeline", "predict", "forecast", "estimate", "project", "ready"]):
            agents.append(AgentType.FORECASTER)
        
        if any(w in query_lower for w in ["how", "fix", "resolve", "action", "recommend", "what should", "solution"]):
            agents.append(AgentType.RESOLVER)
        
        if any(w in query_lower for w in ["email", "notify", "communicate", "message", "draft", "report", "send"]):
            agents.append(AgentType.COMMUNICATOR)
        
        # Default to diagnostic + resolver if unclear
        if not agents:
            agents = [AgentType.DIAGNOSTIC, AgentType.RESOLVER]
        
        return agents
    
    def process(self, state: AgentState) -> AgentState:
        """Process the query and determine routing"""
        
        # Get overall summary for context
        overall_result = self.use_tool("get_overall_summary")
        context_info = overall_result.data if overall_result.success else {}
        
        # Analyze the query
        prompt = f"""Analyze this clinical trial query and determine:
1. What is the user asking for?
2. What type of response do they need?
3. What priority should this have?

Query: {state.user_query}

Current Data Summary:
- Total Patients: {context_info.get('total_patients', 'N/A')}
- Studies: {context_info.get('total_studies', 'N/A')}
- Sites: {context_info.get('total_sites', 'N/A')}
- Avg DQI: {context_info.get('avg_dqi', 'N/A'):.1f}
- Tier 1 Clean Rate: {context_info.get('tier1_clean_rate', 'N/A'):.1f}%
- DB Lock Ready: {context_info.get('db_lock_ready', 'N/A')}

Provide a brief analysis of what the user needs."""
        
        response = self.generate(prompt, state, include_tools=False)
        
        if response.success:
            # Determine agents needed
            agents_needed = self._determine_agents_needed(state.user_query, state)
            
            # Create subtasks
            state.subtasks = [
                {"agent": agent.value, "status": "pending"}
                for agent in agents_needed
            ]
            
            # Set next agent
            if agents_needed:
                state.next_agent = agents_needed[0]
            
            # Store context for other agents
            state.context["overall_summary"] = context_info
            
            # Add analysis to messages
            state.add_message("assistant", response.content, agent=self.agent_type.value)
            
            logger.info(f"Supervisor routing to: {[a.value for a in agents_needed]}")
        else:
            state.errors.append(f"Supervisor analysis failed: {response.error}")
        
        return state


class DiagnosticAgent(BaseAgent):
    """
    Diagnostic Agent - Investigates and analyzes issues
    
    Responsibilities:
    - Investigate data quality issues
    - Form hypotheses with confidence levels
    - Identify root causes
    - Suggest verification steps
    """
    
    def __init__(self, **kwargs):
        kwargs['llm'] = get_shared_llm()
        super().__init__(AgentType.DIAGNOSTIC, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are the Diagnostic Agent for clinical trial data quality.

Your responsibilities:
1. Investigate data quality issues thoroughly
2. Form hypotheses with confidence levels (always express uncertainty)
3. Identify potential root causes
4. Suggest verification steps

Guidelines:
- Never claim certainty, always express confidence as percentages
- Consider multiple possible causes
- Look for patterns across sites/studies
- Cite evidence for your hypotheses
- Be specific about what data supports your conclusions"""
    
    @property
    def allowed_tools(self) -> List[str]:
        return [
            "get_patient", "get_site_summary", "get_study_summary",
            "get_cascade_impact", "search_patterns", "detect_anomalies",
            "get_high_priority_patients"
        ]
    
    def process(self, state: AgentState) -> AgentState:
        """Investigate and diagnose issues"""
        
        context_data = {}
        
        # Check for site-specific queries
        query_lower = state.user_query.lower()
        
        # Try to extract and look up site
        for word in state.user_query.replace(",", " ").replace(".", " ").split():
            if "site" in word.lower() or word.startswith("Site_"):
                site_id = word if word.startswith("Site_") else f"Site_{word.split('_')[-1] if '_' in word else word}"
                result = self.use_tool("get_site_summary", site_id=site_id)
                if result.success and "error" not in result.data:
                    context_data["site"] = result.data
                    break
        
        # Check for study-specific queries
        for word in state.user_query.replace(",", " ").replace(".", " ").split():
            if "study" in word.lower() or word.startswith("Study_"):
                study_id = word if word.startswith("Study_") else f"Study_{word.split('_')[-1] if '_' in word else word}"
                result = self.use_tool("get_study_summary", study_id=study_id)
                if result.success and "error" not in result.data:
                    context_data["study"] = result.data
                    break
        
        # Get high priority patients for context
        high_priority = self.use_tool("get_high_priority_patients", limit=10)
        if high_priority.success:
            context_data["high_priority_patients"] = high_priority.data
        
        # Check for anomalies
        anomalies = self.use_tool("detect_anomalies", limit=5)
        if anomalies.success:
            context_data["anomalies"] = anomalies.data
        
        # Generate diagnostic analysis
        prompt = f"""Analyze this clinical trial situation and provide:
1. Key observations from available data
2. Possible root causes (with confidence %)
3. Hypotheses to investigate
4. Verification steps recommended

Retrieved Data:
{context_data}

Overall Context:
{state.context.get('overall_summary', {})}

Focus on being thorough but acknowledge uncertainty. 
Provide specific, actionable insights."""
        
        response = self.generate(prompt, state)
        
        if response.success:
            # Create hypothesis
            hypothesis = Hypothesis(
                hypothesis_id=f"HYP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                description=response.content[:500],
                confidence=0.75,
                evidence=[Evidence(
                    source="diagnostic_analysis",
                    description="Analysis based on available data",
                    confidence=0.75,
                    data=context_data
                )],
                verification_steps=[
                    "Review site-level data patterns",
                    "Check for similar issues across studies",
                    "Validate with source data if available"
                ]
            )
            
            state.hypotheses.append(hypothesis)
            state.investigation_notes = response.content
            state.add_message("assistant", response.content, agent=self.agent_type.value)
            
            # Route to next agent
            remaining = [s for s in state.subtasks 
                        if s["status"] == "pending" and s["agent"] != self.agent_type.value]
            if remaining:
                state.next_agent = AgentType(remaining[0]["agent"])
            else:
                state.next_agent = None
            
            # Mark this subtask complete
            for subtask in state.subtasks:
                if subtask["agent"] == self.agent_type.value:
                    subtask["status"] = "completed"
        else:
            state.errors.append(f"Diagnostic analysis failed: {response.error}")
        
        return state


class ForecasterAgent(BaseAgent):
    """
    Forecaster Agent - Predicts timelines and outcomes
    
    Responsibilities:
    - Predict timelines with uncertainty bands
    - Estimate resource requirements
    - Project data quality trajectories
    - Identify risks to milestones
    """
    
    def __init__(self, **kwargs):
        kwargs['llm'] = get_shared_llm()
        super().__init__(AgentType.FORECASTER, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are the Forecaster Agent for clinical trial projections.

Your responsibilities:
1. Predict timelines with uncertainty ranges (never point estimates)
2. Estimate resource requirements
3. Project data quality trajectories
4. Identify risks to milestones

Guidelines:
- Always provide ranges: "7-14 days" not "10 days"
- Include confidence intervals
- List assumptions explicitly
- Identify risks that could affect projections
- Be realistic about uncertainty"""
    
    @property
    def allowed_tools(self) -> List[str]:
        return ["get_dblock_projection", "get_site_benchmark", "get_study_summary", "get_overall_summary"]
    
    def process(self, state: AgentState) -> AgentState:
        """Generate forecasts and projections"""
        
        # Get projection data
        dblock_result = self.use_tool("get_dblock_projection")
        overall_result = self.use_tool("get_overall_summary")
        
        projection_data = {
            "dblock": dblock_result.data if dblock_result.success else {},
            "overall": overall_result.data if overall_result.success else {}
        }
        
        prompt = f"""Based on available data, provide forecasts for:
1. Timeline estimates (with ranges, e.g., "2-4 weeks")
2. Resource requirements
3. Risk factors that could delay progress
4. Key milestones and their likelihood

Current Status:
- DB Lock Ready: {projection_data['dblock'].get('ready_now', 'N/A')}
- Ready Rate: {projection_data['dblock'].get('ready_rate', 'N/A'):.1f}%
- Eligible Patients: {projection_data['dblock'].get('eligible_patients', 'N/A')}
- Overall Tier 1 Clean: {projection_data['overall'].get('tier1_clean_rate', 'N/A'):.1f}%

Previous Analysis:
{state.investigation_notes[:500] if state.investigation_notes else 'None'}

Always express uncertainty with ranges, not point estimates.
List key assumptions that underlie your projections."""
        
        response = self.generate(prompt, state)
        
        if response.success:
            forecast = Forecast(
                metric="db_lock_readiness",
                prediction=projection_data['dblock'].get('ready_rate', 50) / 100,
                lower_bound=max(0, projection_data['dblock'].get('ready_rate', 50) / 100 - 0.15),
                upper_bound=min(1, projection_data['dblock'].get('ready_rate', 50) / 100 + 0.20),
                confidence=0.7,
                timeframe="14-28 days",
                assumptions=[
                    "Current resolution rate continues",
                    "No major holidays or site closures",
                    "Resources remain stable"
                ],
                risks=[
                    "Site coordinator availability",
                    "PI signature delays",
                    "Unexpected data quality issues"
                ]
            )
            
            state.forecasts.append(forecast)
            state.timeline_projection = response.content
            state.add_message("assistant", response.content, agent=self.agent_type.value)
            
            # Route to next agent
            remaining = [s for s in state.subtasks 
                        if s["status"] == "pending" and s["agent"] != self.agent_type.value]
            if remaining:
                state.next_agent = AgentType(remaining[0]["agent"])
            else:
                state.next_agent = None
            
            for subtask in state.subtasks:
                if subtask["agent"] == self.agent_type.value:
                    subtask["status"] = "completed"
        else:
            state.errors.append(f"Forecasting failed: {response.error}")
        
        return state


class ResolverAgent(BaseAgent):
    """
    Resolver Agent - Creates action plans and recommendations
    
    Responsibilities:
    - Recommend solutions based on issue type
    - Prioritize by impact and urgency
    - Estimate effort required
    - Consider regulatory requirements
    """
    
    def __init__(self, **kwargs):
        kwargs['llm'] = get_shared_llm()
        super().__init__(AgentType.RESOLVER, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are the Resolver Agent for clinical trial issue resolution.

Your responsibilities:
1. Recommend specific, actionable solutions
2. Prioritize by impact and urgency
3. Estimate effort required for each action
4. Consider regulatory requirements

Guidelines:
- Be specific: "Contact Site Coordinator to batch sign 5 pending CRFs" not "Fix signatures"
- Assign to roles: CRA, Data Manager, Site Coordinator, etc.
- Estimate hours for each task
- Consider cascade effects - which fixes unlock other issues
- Flag any regulatory considerations"""
    
    @property
    def allowed_tools(self) -> List[str]:
        return ["search_resolutions", "get_cascade_impact", "create_task", "get_high_priority_patients"]
    
    def process(self, state: AgentState) -> AgentState:
        """Generate action recommendations"""
        
        # Search for relevant resolutions
        resolution_result = self.use_tool("search_resolutions", issue_type="open_queries")
        high_priority = self.use_tool("get_high_priority_patients", limit=5)
        
        resolution_context = {
            "templates": resolution_result.data if resolution_result.success else [],
            "high_priority": high_priority.data if high_priority.success else []
        }
        
        prompt = f"""Based on the diagnostic findings and forecasts, create a prioritized action plan.

Diagnostic Notes:
{state.investigation_notes[:800] if state.investigation_notes else 'None'}

Forecasts:
{state.timeline_projection[:500] if state.timeline_projection else 'None'}

High Priority Patients:
{resolution_context['high_priority'][:5] if resolution_context['high_priority'] else 'None'}

For each recommended action, specify:
1. Specific action to take (be detailed)
2. Who is responsible (CRA, Data Manager, Site, etc.)
3. Priority (Critical/High/Medium/Low)
4. Estimated effort (hours)
5. Expected impact (what gets unblocked)
6. Any dependencies or prerequisites

Order actions by impact - what should be done first for maximum benefit."""
        
        response = self.generate(prompt, state)
        
        if response.success:
            recommendation = Recommendation(
                recommendation_id=f"REC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                action="Review and resolve priority data quality items",
                priority=TaskPriority.HIGH,
                confidence=0.8,
                impact="Improve clean patient rate and unblock DB Lock",
                effort_hours=4.0,
                responsible_role="Data Manager / CRA",
                rationale=response.content[:500],
                steps=[
                    "Review high priority patient list",
                    "Categorize issues by type",
                    "Resolve in cascade order",
                    "Verify resolution"
                ],
                requires_approval=True
            )
            
            state.recommendations.append(recommendation)
            state.action_plan = response.content
            state.add_message("assistant", response.content, agent=self.agent_type.value)
            
            # Route to next agent
            remaining = [s for s in state.subtasks 
                        if s["status"] == "pending" and s["agent"] != self.agent_type.value]
            if remaining:
                state.next_agent = AgentType(remaining[0]["agent"])
            else:
                state.next_agent = None
            
            for subtask in state.subtasks:
                if subtask["agent"] == self.agent_type.value:
                    subtask["status"] = "completed"
        else:
            state.errors.append(f"Resolution planning failed: {response.error}")
        
        return state


class CommunicatorAgent(BaseAgent):
    """
    Communicator Agent - Drafts messages and notifications
    
    Responsibilities:
    - Draft clear, professional messages
    - Tailor tone to the recipient
    - Include relevant context
    - Suggest appropriate channels
    """
    
    def __init__(self, **kwargs):
        kwargs['llm'] = get_shared_llm()
        super().__init__(AgentType.COMMUNICATOR, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are the Communicator Agent for clinical trial communications.

Your responsibilities:
1. Draft clear, professional messages
2. Tailor tone to the recipient (site staff, sponsor, CRA)
3. Include relevant context without overwhelming
4. Suggest appropriate channels (email, notification, report)

Guidelines:
- Be concise but complete
- Use appropriate urgency indicators
- Include specific action items with deadlines
- Never auto-send - always draft for human review
- Mark all communications as DRAFT"""
    
    @property
    def allowed_tools(self) -> List[str]:
        return ["draft_query_email", "create_task"]
    
    def process(self, state: AgentState) -> AgentState:
        """Draft communications"""
        
        prompt = f"""Draft appropriate communication based on the analysis and recommendations.

Action Plan Summary:
{state.action_plan[:800] if state.action_plan else 'None available'}

Recommendations: {len(state.recommendations)} items

Create a professional message that:
1. Summarizes the key findings
2. Lists specific action items with responsible parties
3. Provides clear deadlines
4. Uses appropriate tone for clinical trial context

Format as a ready-to-send email or notification.
Mark as [DRAFT - REQUIRES HUMAN APPROVAL]"""
        
        response = self.generate(prompt, state)
        
        if response.success:
            comm = Communication(
                communication_id=f"COMM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                type="email",
                recipient="Site Coordinator / Study Team",
                recipient_role="Site",
                subject="Action Required: Data Quality Items - DRAFT",
                body=response.content,
                priority=TaskPriority.MEDIUM,
                requires_approval=True
            )
            
            state.communications.append(comm)
            state.notifications_queued += 1
            state.add_message("assistant", response.content, agent=self.agent_type.value)
            
            # Mark complete
            for subtask in state.subtasks:
                if subtask["agent"] == self.agent_type.value:
                    subtask["status"] = "completed"
            
            state.next_agent = None
        else:
            state.errors.append(f"Communication drafting failed: {response.error}")
        
        return state


class SynthesizerAgent(BaseAgent):
    """
    Synthesizer Agent - Combines outputs into final response
    
    This is a special agent that runs at the end to create the final response.
    """
    
    def __init__(self, **kwargs):
        kwargs['llm'] = get_shared_llm()
        super().__init__(AgentType.SUPERVISOR, **kwargs)
        self.name = "Synthesizer"
    
    @property
    def system_prompt(self) -> str:
        return """You are the Synthesizer Agent. Your job is to combine all agent outputs into a clear, actionable final response.

Create a response that:
1. Directly answers the user's original question
2. Summarizes key findings concisely
3. Lists recommended actions in priority order
4. Highlights any items requiring human approval
5. Is professional, clear, and actionable

Structure your response with clear sections:
- Summary (2-3 sentences)
- Key Findings
- Recommended Actions
- Next Steps"""
    
    def process(self, state: AgentState) -> AgentState:
        """Synthesize final response"""
        
        prompt = f"""Synthesize all agent outputs into a final response for the user.

Original Query: {state.user_query}

Diagnostic Findings:
{state.investigation_notes[:600] if state.investigation_notes else 'None'}

Forecasts:
{state.timeline_projection[:400] if state.timeline_projection else 'None'}

Action Plan:
{state.action_plan[:600] if state.action_plan else 'None'}

Recommendations Generated: {len(state.recommendations)}
Communications Drafted: {len(state.communications)}
Hypotheses Formed: {len(state.hypotheses)}

Create a clear, well-structured response that directly addresses what the user asked.
Be concise but complete. Prioritize actionable information."""
        
        response = self.generate(prompt, state)
        
        if response.success:
            state.final_response = response.content
            state.summary = response.content[:500]
            state.task_status = TaskStatus.COMPLETED
        else:
            # Fallback response
            state.final_response = f"""Based on the analysis:

**Summary:**
{state.investigation_notes[:300] if state.investigation_notes else 'Analysis completed.'}

**Recommendations:**
{state.action_plan[:300] if state.action_plan else 'See detailed recommendations.'}

**Next Steps:**
- Review the recommendations above
- Approve any pending communications
- Implement priority actions

Note: {len(state.recommendations)} recommendations generated, {len(state.communications)} communications drafted."""
            state.task_status = TaskStatus.COMPLETED
        
        return state


# ============================================================
# ORCHESTRATOR
# ============================================================

class AgentOrchestrator:
    """
    Orchestrates the multi-agent workflow using LangGraph.
    
    The workflow:
    1. Supervisor analyzes query and routes to specialists
    2. Specialists process in sequence (diagnostic → forecaster → resolver → communicator)
    3. Synthesizer combines outputs into final response
    """
    
    def __init__(self):
        """Initialize the orchestrator"""
        # Initialize agents (they share the LLM instance)
        self.supervisor = SupervisorAgent()
        self.diagnostic = DiagnosticAgent()
        self.forecaster = ForecasterAgent()
        self.resolver = ResolverAgent()
        self.communicator = CommunicatorAgent()
        self.synthesizer = SynthesizerAgent()
        
        # Build the graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info("AgentOrchestrator initialized with shared LLM")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph with AgentState
        graph = StateGraph(AgentState)
        
        # Add nodes for each agent
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("diagnostic", self.diagnostic)
        graph.add_node("forecaster", self.forecaster)
        graph.add_node("resolver", self.resolver)
        graph.add_node("communicator", self.communicator)
        graph.add_node("synthesizer", self.synthesizer)
        
        # Define routing logic
        def route_from_supervisor(state: AgentState) -> str:
            """Route from supervisor to first specialist"""
            if state.next_agent:
                return state.next_agent.value
            return "synthesizer"
        
        def route_next(state: AgentState) -> str:
            """Route to next agent or synthesizer"""
            if state.next_agent:
                return state.next_agent.value
            return "synthesizer"
        
        # Set entry point
        graph.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "diagnostic": "diagnostic",
                "forecaster": "forecaster",
                "resolver": "resolver",
                "communicator": "communicator",
                "synthesizer": "synthesizer"
            }
        )
        
        # Add conditional edges from each specialist
        for agent in ["diagnostic", "forecaster", "resolver", "communicator"]:
            graph.add_conditional_edges(
                agent,
                route_next,
                {
                    "diagnostic": "diagnostic",
                    "forecaster": "forecaster",
                    "resolver": "resolver",
                    "communicator": "communicator",
                    "synthesizer": "synthesizer"
                }
            )
        
        # Synthesizer goes to END
        graph.add_edge("synthesizer", END)
        
        return graph
    
    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        thread_id: Optional[str] = None
    ) -> AgentState:
        """
        Run the agent workflow.
        
        Args:
            query: User query to process
            context: Optional context data
            priority: Task priority
            thread_id: Optional thread ID for conversation memory
            
        Returns:
            Final AgentState with all outputs
        """
        # Create initial state
        state = create_initial_state(query, context, priority)
        
        # Configure thread
        config = {"configurable": {"thread_id": thread_id or state.task_id}}
        
        logger.info(f"Starting orchestration for: {query[:50]}...")
        
        # Run the graph
        final_state_dict = None
        for step in self.app.stream(state, config):
            # Log progress
            for node_name, node_state in step.items():
                logger.info(f"  → {node_name} completed")
                final_state_dict = node_state
        
        # Convert dict back to AgentState (LangGraph returns dict)
        if final_state_dict is not None:
            if isinstance(final_state_dict, dict):
                final_state = AgentState(**final_state_dict)
            else:
                final_state = final_state_dict
        else:
            final_state = state
            final_state.task_status = TaskStatus.FAILED
            final_state.errors.append("No output from workflow")
        
        logger.info(f"Orchestration complete. Status: {final_state.task_status.value}")
        
        return final_state
    
    def run_simple(self, query: str) -> str:
        """
        Simple interface - just returns the final response string.
        
        Args:
            query: User query
            
        Returns:
            Final response string
        """
        state = self.run(query)
        return state.final_response


# Convenience function
def get_orchestrator() -> AgentOrchestrator:
    """Get the agent orchestrator instance"""
    return AgentOrchestrator()