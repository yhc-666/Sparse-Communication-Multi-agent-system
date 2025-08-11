# AgentVerse Multi-Agent System: Complete Rules & Memory Tutorial

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [The Rule System: Orchestrating Agent Interactions](#the-rule-system)
3. [Memory System: Managing Conversation History](#memory-system)
4. [Complete Execution Flow](#complete-execution-flow)
5. [Configuration Guide](#configuration-guide)
6. [Practical Examples](#practical-examples)
7. [Troubleshooting & Best Practices](#troubleshooting--best-practices)

---

## System Architecture Overview

AgentVerse is a multi-agent conversation framework where agents interact through a structured environment controlled by **Rules** and maintain conversation history through **Memory** systems.

### Core Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                       Environment                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                      Rule System                    │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │    │
│  │  │  Order   │ │Visibility│ │ Selector │           │    │
│  │  └──────────┘ └──────────┘ └──────────┘           │    │
│  │  ┌──────────┐ ┌──────────┐                        │    │
│  │  │ Updater  │ │Describer │                        │    │
│  │  └──────────┘ └──────────┘                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent N │       │
│  │ +Memory │  │ +Memory │  │ +Memory │  │ +Memory │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

- **Environment**: Manages the overall conversation flow and turn-taking
- **Rule System**: Five components that control how agents interact
- **Agents**: Individual participants with their own memory and logic
- **Memory**: Stores conversation history for context

---

## The Rule System

The Rule system (`agentverse/environments/rules/`) consists of five interconnected components that orchestrate multi-agent conversations.

### Rule Base Class
```python
# agentverse/environments/rules/base.py
class Rule:
    def __init__(self, order, visibility, selector, updater, describer):
        self.order = OrderRegistry.build(**order)
        self.visibility = VisibilityRegistry.build(**visibility)
        self.selector = SelectorRegistry.build(**selector)
        self.updater = UpdaterRegistry.build(**updater)
        self.describer = DescriberRegistry.build(**describer)
```

### 1. Order Rules: Who Speaks When?

Order rules determine which agents can speak in each turn.

#### **Sequential Order** (`order/sequential.py`)
```python
@OrderRegistry.register("sequential")
class SequentialOrder(BaseOrder):
    """Agents speak one at a time in order"""
    
    def get_next_agent_idx(self, environment) -> List[int]:
        # Round-robin: Agent 0, 1, 2, 0, 1, 2...
        return [(environment.cnt_turn) % len(environment.agents)]
```

**Use Case**: Traditional turn-taking debates

#### **Concurrent Order** (`order/concurrent.py`)
```python
@OrderRegistry.register("concurrent")
class ConcurrentOrder(BaseOrder):
    """All agents speak simultaneously"""
    
    def get_next_agent_idx(self, environment) -> List[int]:
        return list(range(len(environment.agents)))  # [0, 1, 2, ...]
```

**Use Case**: Parallel reasoning, brainstorming

#### **Random Order** (`order/random.py`)
```python
@OrderRegistry.register("random")
class RandomOrder(BaseOrder):
    """Random agent speaks each turn"""
    
    def get_next_agent_idx(self, environment) -> List[int]:
        return [random.randint(0, len(environment.agents) - 1)]
```

**Use Case**: Unpredictable discussions

### 2. Visibility Rules: Who Sees Whom?

Visibility rules control which agents are visible to each other.

#### **All Visibility** (`visibility/all.py`)
```python
@VisibilityRegistry.register("all")
class AllVisibility(BaseVisibility):
    """Every agent sees all other agents"""
    
    def update_visible_agents(self, environment):
        for agent in environment.agents:
            agent.set_receiver(set(a.name for a in environment.agents))
```

**Use Case**: Open debates where everyone sees everyone

#### **Oneself Visibility** (`visibility/oneself.py`)
```python
@VisibilityRegistry.register("oneself")
class OneselfVisibility(BaseVisibility):
    """Agents only see themselves"""
    
    def update_visible_agents(self, environment):
        for agent in environment.agents:
            agent.set_receiver({agent.name})
```

**Use Case**: Independent reasoning without influence

### 3. Selector Rules: Which Messages Matter?

Selector rules filter which messages are kept from each turn.

#### **Basic Selector** (`selector/basic.py`)
```python
@SelectorRegistry.register("basic")
class BasicSelector(BaseSelector):
    """Keep all non-empty messages"""
    
    def select_message(self, environment, messages) -> List[Message]:
        return [m for m in messages if m.content != ""]
```

**Use Case**: Standard conversations keeping all contributions

### 4. Updater Rules: How Memory Gets Updated

Updater rules distribute messages to agent memories.

#### **Basic Updater** (`updater/basic.py`)
```python
@UpdaterRegistry.register("basic")
class BasicUpdater(BaseUpdater):
    """Distribute messages based on receiver field"""
    
    def update_memory(self, environment):
        for message in environment.last_messages:
            if "all" in message.receiver:
                # Broadcast to everyone
                for agent in environment.agents:
                    agent.add_message_to_memory([message])
            else:
                # Send to specific receivers
                for agent in environment.agents:
                    if agent.name in message.receiver:
                        agent.add_message_to_memory([message])
```

**Key Feature**: Handles both broadcast ("all") and targeted messaging

### 5. Describer Rules: How Environment is Described

Describer rules generate environment descriptions for agents.

#### **Basic Describer** (`describer/basic.py`)
```python
@DescriberRegistry.register("basic")
class BasicDescriber(BaseDescriber):
    """Simple turn counter description"""
    
    def get_env_description(self, environment) -> List[str]:
        return [f"Turn {environment.cnt_turn}/{environment.max_turns}"] * len(environment.agents)
```

---

## Memory System

The memory system stores and manages conversation history for each agent.

### Memory Architecture

```python
# Base Memory Class
class BaseMemory(BaseModel):
    @abstractmethod
    def add_message(self, messages: List[Message]) -> None
    
    @abstractmethod
    def to_string(self, agent_name: str = None) -> str
    
    @abstractmethod
    def reset(self) -> None
```

### ChatHistoryMemory (Default Implementation)

```python
@memory_registry.register("chat_history")
class ChatHistoryMemory(BaseMemory):
    messages: List[Message] = []
    
    def add_message(self, messages: List[Message]) -> None:
        """Append messages to history"""
        self.messages.extend(messages)
    
    def to_string(self, agent_name: str = None) -> str:
        """Format as 'Speaker: content' lines"""
        return "\n".join([
            f"{m.sender}: {m.content}"
            for m in self.messages
            if m.content and m.content != "[Silence]"
        ])
```

### Message Structure

```python
class Message(BaseModel):
    content: str = ""                    # Message text
    sender: str = ""                     # Who sent it
    receiver: Set[str] = {"all"}        # Who receives it
    tool_response: List[Tuple] = []     # Optional tool outputs
```

### Memory Flow

1. **Agent speaks** → Creates Message with content and receivers
2. **Environment collects** → Gathers all messages from turn
3. **Updater distributes** → Routes messages to agent memories
4. **Memory stores** → Each agent adds messages to their history
5. **Next turn** → Agents use memory.to_string() for context

---

## Complete Execution Flow

Here's what happens during one turn of conversation:

### Step-by-Step Turn Execution

```python
# In BasicEnvironment.step()
async def step(self) -> List[Message]:
    # 1. ORDER: Determine who speaks this turn
    agent_ids = self.rule.get_next_agent_idx(self)
    # Example: [0, 2] means agents 0 and 2 speak
    
    # 2. DESCRIBER: Generate environment descriptions
    env_descriptions = self.rule.get_env_description(self)
    # Example: ["Turn 3/10", "Turn 3/10", ...]
    
    # 3. AGENTS SPEAK: Collect responses
    messages = await asyncio.gather(
        *[self.agents[i].astep(env_descriptions[i]) for i in agent_ids]
    )
    
    # 4. SELECTOR: Filter messages
    selected_messages = self.rule.select_message(self, messages)
    self.last_messages = selected_messages
    
    # 5. UPDATER: Distribute to memories
    self.rule.update_memory(self)
    # Each agent's memory now contains the new messages
    
    # 6. VISIBILITY: Update who can see whom
    self.rule.update_visible_agents(self)
    
    # 7. Increment turn counter
    self.cnt_turn += 1
    
    return selected_messages
```

### Memory Update Detail

When `rule.update_memory(self)` is called:

```python
# In BasicUpdater.update_memory()
for message in environment.last_messages:
    if "all" in message.receiver:
        # Broadcast mode - everyone gets this message
        for agent in environment.agents:
            agent.add_message_to_memory([message])
    else:
        # Targeted mode - only specified receivers
        for agent in environment.agents:
            if agent.name in message.receiver:
                agent.add_message_to_memory([message])

# If no one spoke, add silence marker
if not any_messages:
    for agent in environment.agents:
        agent.add_message_to_memory([Message(content="[Silence]")])
```

### Agent Memory Usage

When an agent needs to respond:

```python
# In agent.step()
def step(self, env_description: str) -> Message:
    # 1. Get conversation history from memory
    chat_history = self.memory.to_string(self.name)
    
    # 2. Build prompt with history
    prompt = self.prompt_template.replace("${chat_history}", chat_history)
    
    # 3. Generate response using LLM
    response = self.llm.generate_response(prompt)
    
    # 4. Create message with receivers
    return Message(
        content=response,
        sender=self.name,
        receiver=self.get_receiver()  # Set by visibility rule
    )
```

---

## Configuration Guide

### YAML Configuration Structure

```yaml
# Example from final_debate_config.yaml
environment:
  env_type: llm_eval
  max_turns: 15
  rule:
    order:
      type: sequential  # or concurrent, random
    visibility:
      type: all        # or oneself, llmeval_blind_judge
    selector:
      type: basic      # Currently only basic available
    updater:
      type: basic      # Currently only basic available
    describer:
      type: basic      # Currently only basic available

agents:
  - agent_type: final_debate_multi
    name: "LP supporter"
    memory:
      memory_type: chat_history  # or summary, vectorstore
    memory_manipulator:
      memory_manipulator_type: basic  # or summary, reflection
    prompt_template: |
      You are ${agent_name}.
      Context: ${context}
      Question: ${question}
      
      Conversation history:
      ${chat_history}
      
      Your response:
```

### Common Configuration Patterns

#### Pattern 1: Sequential Debate
```yaml
rule:
  order:
    type: sequential  # Agents take turns
  visibility:
    type: all        # Everyone sees all messages
```
**Result**: Traditional debate with turn-taking

#### Pattern 2: Parallel Brainstorming
```yaml
rule:
  order:
    type: concurrent  # All speak at once
  visibility:
    type: all        # Everyone sees all messages
```
**Result**: Simultaneous idea generation

#### Pattern 3: Independent Analysis
```yaml
rule:
  order:
    type: concurrent  # All speak at once
  visibility:
    type: oneself    # Agents only see their own messages
```
**Result**: Parallel independent reasoning

---

## Practical Examples

### Example 1: Three-Agent Debate

**Scenario**: LP, FOL, and SAT supporters debate a logic problem

**Configuration**:
```yaml
environment:
  max_turns: 15
  rule:
    order:
      type: sequential
    visibility:
      type: all
```

**Execution Flow**:
```
Turn 1: LP supporter speaks → Message sent to all → All memories updated
Turn 2: FOL supporter speaks → Message sent to all → All memories updated
Turn 3: SAT supporter speaks → Message sent to all → All memories updated
Turn 4: LP supporter speaks (cycle repeats)...
```

**Memory State After Turn 3**:
```
All agents have identical memory:
- Message 1: "LP supporter: I believe logic programming..."
- Message 2: "FOL supporter: First-order logic provides..."
- Message 3: "SAT supporter: The Z3 solver demonstrates..."
```

### Example 2: Parallel Independent Solving

**Configuration**:
```yaml
environment:
  max_turns: 1
  rule:
    order:
      type: concurrent
    visibility:
      type: oneself
```

**Execution Flow**:
```
Turn 1: All agents speak simultaneously
- Each agent only receives their own message
- No cross-contamination of ideas
```

**Memory State After Turn 1**:
```
Agent 1 memory: Only their own message
Agent 2 memory: Only their own message
Agent 3 memory: Only their own message
```

---

## Troubleshooting & Best Practices

### Common Issues

#### Issue 1: Memory Manipulators Not Working
**Problem**: Configured memory manipulators don't execute
**Cause**: Most agents don't call `manipulate_memory()`
**Solution**: Use `llm_eval_multi_agent_con` or fix the agent implementation

#### Issue 2: Memory Growing Too Large
**Problem**: Token limits exceeded in long conversations
**Solution**: 
```yaml
memory:
  memory_type: summary
  buffer_size: 30
memory_manipulator:
  memory_manipulator_type: summary
```

#### Issue 3: Agents Not Receiving Messages
**Problem**: Messages not appearing in agent memory
**Check**:
1. Message.receiver field includes target agent
2. Visibility rule sets correct receivers
3. Updater rule is "basic"

### Best Practices

1. **For Debates**: Use sequential order + all visibility
2. **For Brainstorming**: Use concurrent order + all visibility
3. **For Independent Analysis**: Use concurrent order + oneself visibility
4. **For Long Conversations**: Use summary memory + summary manipulator
5. **For Testing**: Start with basic everything, add complexity gradually

### Memory Synchronization

Remember: With default settings (`receiver="all"`), all agents maintain **identical memory**. This ensures:
- Fair debates with equal information
- Coherent conversation flow
- No information asymmetry

### Performance Tips

1. **Limit max_turns** to prevent memory explosion
2. **Use summary memory** for conversations > 50 turns
3. **Configure buffer_size** based on token limits
4. **Monitor memory size** with `len(agent.memory.to_string())`

---

## Conclusion

The AgentVerse system provides a flexible framework for multi-agent conversations through:

- **Rules** that orchestrate agent interactions
- **Memory** that maintains conversation context
- **Configuration** that adapts to different scenarios

Understanding how rules and memory work together is key to building effective multi-agent systems. The current implementation provides a solid foundation, though some features (like memory manipulators) need fixes to work properly.

For questions or contributions, refer to the codebase at `/agentverse/` and experiment with different configurations to understand the system behavior.