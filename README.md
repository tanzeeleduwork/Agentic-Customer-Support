# Support Ticket Triage Agent - AI Agent Implementation

An AI agent that automates support ticket handling using LLM-powered tool use. Given a raw support ticket, the agent classifies urgency, extracts structured information, routes to the correct team, and drafts a professional first response — all autonomously in under 10 seconds.

## Project Overview

This project builds a working AI agent for automated support ticket triage using agentic tool use and structured LLM outputs. The agent processes a ticket through four sequential tools and produces a **complete triage result** including urgency classification, team routing, and a ready-to-send draft response.

## Key Results

| Metric | Value |
|--------|-------|
| **LLM Backend** | Groq API (llama-3.3-70b-versatile) |
| **Tools Implemented** | 4 (classify, extract, route, draft) |
| **Average Processing Time** | Under 10 seconds per ticket |
| **Output Format** | Structured JSON + printed triage report |
| **Ticket Categories Handled** | 5 (accessibility, technical, billing, account, general) |
| **Urgency Levels** | 4 (critical, high, medium, low) |

## Agent Architecture

```
Raw Support Ticket (plain text)
            |
            v
+--------------------------------+
|     System Prompt + Context    |
|   (Transreport domain knowledge)|
+--------------------------------+
            |
            v
+--------------------------------+
|        Groq LLM Backend        |
|   llama-3.3-70b-versatile      |
|                                |
|   Agentic Loop:                |
|   1. classify_ticket    -----> | urgency + category + confidence
|   2. extract_ticket_info ----> | issue summary + sentiment
|   3. route_ticket       -----> | team + SLA + escalation flag
|   4. draft_response     -----> | subject line + email body
+--------------------------------+
            |
            v
  Triage Report (stdout) + JSON file
```

The agent uses **LLM tool use (function calling)** — the model autonomously decides which tools to call and in what order, making it genuinely agentic rather than a hardcoded pipeline.

## Project Workflow

### 1. Tool Design
Defined four structured tools using the OpenAI-compatible function calling schema, each with a precise JSON schema enforcing output structure:

- **classify_ticket** — determines urgency level and category with a confidence score
- **extract_ticket_info** — extracts core issue, affected service, user sentiment, and human review flag
- **route_ticket** — assigns the correct internal team and sets an SLA target in hours
- **draft_response** — generates a professional, empathetic first-response email

### 2. Agentic Loop Implementation
Built a `while True` loop that continues calling the LLM until `stop_reason` signals no further tool calls. Full conversation history is maintained across turns so the model retains context from previous tool results.

### 3. Prompt Engineering
Designed a domain-specific system prompt that:
- Establishes the agent's role within Transreport's operational context
- Enforces tool call order explicitly
- Instructs the model to prioritise empathy for users with accessibility needs

### 4. Structured Output Handling
Each tool call response is parsed from JSON, validated, and stored in a typed result dictionary. Results are both printed as a readable triage report and persisted to a JSON file for downstream use.

### 5. Production Extension Points
The `handle_tool_call` function is designed as a drop-in integration point. In a production deployment, each tool would call a real service:

| Tool | Production Integration |
|------|----------------------|
| classify_ticket | Internal ML classifier or rules engine |
| extract_ticket_info | CRM lookup or NLP pipeline |
| route_ticket | Ticketing system (Jira, Zendesk) |
| draft_response | Email dispatch via SendGrid |

## Sample Output

**Input ticket:**
```
Hi, I am a wheelchair user and I booked rail assistance for my journey tomorrow
from Manchester Piccadilly to London Euston. The app is showing my booking as
'pending' for the last 3 hours and I cannot confirm if the assistance has been
arranged. I am really worried as I cannot travel without it.
```

**Agent output:**
```
CLASSIFICATION
  Urgency    : HIGH
  Category   : accessibility
  Confidence : 90%

EXTRACTED INFO
  Issue        : Rail assistance booking stuck in pending state before time-sensitive journey
  Affected     : Assistance booking system
  Sentiment    : urgent
  Human review : Required

ROUTING
  Team     : accessibility_team
  SLA      : 2 hours
  Escalate : Yes

DRAFT RESPONSE
  Subject : Re: Your Rail Assistance Booking - We Are On It

  Dear passenger, I completely understand how important this is for your journey
  tomorrow. I have escalated your booking to our accessibility team as a priority
  and we will confirm your assistance arrangements within the next 2 hours...
```

## Technical Stack

- **LLM Provider**: Groq API
- **Model**: llama-3.3-70b-versatile
- **Language**: Python 3.12
- **Core Library**: groq (official Python SDK)
- **Tool Schema**: OpenAI-compatible function calling format
- **Output**: JSON + stdout report

## Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/support-ticket-agent-grok
cd support-ticket-agent-grok

# 2. Install dependencies
pip install groq

# 3. Set your Groq API key (free at console.groq.com)
# PowerShell:
$env:GROQ_API_KEY="your-key-here"
# bash/zsh:
export GROQ_API_KEY="your-key-here"

# 4. Run the demo
python agent_groq.py

# 5. Or triage a custom ticket
python agent_groq.py "My booking is stuck in pending and I travel tomorrow."
```

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| LLM tool use / function calling | 4 tools with enforced JSON schemas |
| Agentic loop | while loop with stop_reason handling |
| Prompt engineering | Domain-specific system prompt with role and constraints |
| Structured outputs | JSON schema enforcement via tool definitions |
| Conversation history | Full message history maintained across tool turns |
| Workflow automation | End-to-end ticket to triage report pipeline |

## Why Tool Use Over Simple Prompting

A naive approach would ask the LLM to classify and respond in a single prompt. Tool use is architecturally superior because:

1. **Guaranteed structure** — JSON schema enforcement means no output parsing failures
2. **Auditability** — every decision is a discrete, logged tool call
3. **Separation of concerns** — each tool does one thing and does it well
4. **Extensibility** — adding a new step means adding one tool definition, not rewriting a prompt
5. **Production-ready** — tool handlers can be swapped for real API calls with no changes to agent logic

## Sample Tickets

Five realistic ticket examples are included in `sample_tickets.json`, covering all supported categories:

- **TKT-001** — Accessibility: urgent rail assistance booking issue
- **TKT-002** — Billing: duplicate subscription charge
- **TKT-003** — Technical: production API returning 503 errors
- **TKT-004** — Account: new user onboarding with guide dog requirement
- **TKT-005** — General: bulk staff account import request

## Limitations and Future Work

- Currently processes one ticket at a time; batch processing loop can be added
- Tool handlers are simulated; production deployment requires real service integrations
- No persistent memory across tickets; a vector store could enable pattern detection
- Confidence thresholds could trigger automatic vs. human-reviewed routing
- A web interface or Slack integration would make the agent directly usable by support teams

## Code Quality Standards

- Clean, well-documented code with descriptive variable names
- Proper section headers separating imports, tools, agent loop, and utilities
- Comprehensive docstrings for all functions with Args and Returns
- Inline comments explaining architectural decisions, not just syntax
- Production extension points clearly marked for real-world deployment

## Conclusion

This project demonstrates a practical end-to-end AI agent implementation: from tool design through agentic loop construction to structured output handling. The use of function calling over simple prompting results in a reliable, auditable, and extensible triage pipeline that can be dropped into a real support workflow with minimal modification.

---

**Project Status**: Complete and Tested

**Author**: Tanzeel Ahmed
**Institution**: Bangladesh University of Engineering and Technology (BUET)
**Year**: 2026
