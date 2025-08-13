from __future__ import annotations

from typing import TYPE_CHECKING, List
import logging

from . import updater_registry as UpdaterRegistry
from .base import BaseUpdater
from agentverse.message import Message

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment
    from agentverse.agents import BaseAgent


@UpdaterRegistry.register("sparse")
class SparseUpdater(BaseUpdater):
    """
    Sparse updater that selectively updates agent memories based on visibility gates.
    Works in conjunction with SparseVisibility to implement selective communication.
    """
    
    def update_memory(self, environment: BaseEnvironment) -> None:
        """
        Update agent memories based on sparse visibility gates.
        Only allows agents to see messages from other agents where the gate is open (O_i→s = 1).
        
        Args:
            environment: The environment containing agents and messages
        """
        # Get gates from visibility rule
        visibility_rule = environment.rule.visibility
        
        # Check if this is sparse visibility
        if not hasattr(visibility_rule, 'gates'):
            logging.warning("No sparse gates found, falling back to basic updater behavior")
            super().update_memory(environment)
            return
        
        # The visibility rule just calculated gates for current round
        round = environment.cnt_turn
        
        # Check if gates exist for this round
        if round not in visibility_rule.gates:
            logging.error(f"Gates not found for round {round}, this shouldn't happen!")
            super().update_memory(environment)
            return
            
        # Use the gates that were just calculated
        gates = visibility_rule.gates[round]
            
        n_agents = len(environment.agents)
        
        # Build agent name to index mapping
        agent_name_to_idx = {}
        for idx, agent in enumerate(environment.agents):
            agent_name_to_idx[agent.name] = idx
        
        # Track if any messages were added
        added = False
        
        # Process each message
        for message in environment.last_messages:
            # Skip empty messages and tool responses
            if len(message.tool_response) > 0:
                self.add_tool_response(
                    message.sender, environment.agents, message.tool_response
                )
                continue
            
            if message.content == "":
                continue
            
            # Get sender index
            if message.sender not in agent_name_to_idx:
                logging.warning(f"Unknown sender: {message.sender}")
                continue
            
            sender_idx = agent_name_to_idx[message.sender]
            
            # Selective memory update based on gates
            for receiver_idx, agent in enumerate(environment.agents):
                # Check if gate is open from sender to receiver
                if gates[sender_idx, receiver_idx] == 1:
                    # Agent can see this message
                    agent.add_message_to_memory([message])
                    added = True
                    logging.info(f"Round {round}: {agent.name} can see message from {message.sender} (gate open)")
                else:
                    # Gate is closed, agent cannot see this message
                    logging.info(f"Round {round}: {agent.name} blocked from {message.sender} (gate closed)")
        
        # If no messages were added (all gates closed), add silence message
        if not added:
            for agent in environment.agents:
                agent.add_message_to_memory([Message(content="[Silence - No visible messages]")])
            logging.info("All agents see silence due to closed gates")
    
    def add_tool_response(
        self,
        name: str,
        agents: List[BaseAgent],
        tool_response: List[str],
    ) -> None:
        """
        Add tool response to the appropriate agent's memory.
        
        Args:
            name: Name of the agent who made the tool call
            agents: List of all agents
            tool_response: The tool response to add
        """
        for agent in agents:
            if agent.name != name:
                continue
            if agent.tool_memory is not None:
                agent.tool_memory.add_message(tool_response)
            break
    
    def reset(self) -> None:
        """Reset updater state for new instance"""
        pass  # No state to reset for sparse updater