"""
Support Ticket Triage Agent
============================
Author      : Tanzeel Ahmed
Description : An AI agent that automates support ticket triage using LLM-powered
              tool use. Given a raw support ticket, the agent classifies urgency,
              extracts structured information, routes to the correct team, and
              drafts a professional first response -- all autonomously.

Architecture:
    - LLM Backend  : Groq API (llama-3.3-70b-versatile)
    - Tool Use     : Four structured tools called sequentially by the model
    - Agent Loop   : Runs until the model stops issuing tool calls
    - Output       : Printed triage report + saved JSON result

Usage:
    # Run demo with built-in sample ticket:
    python agent_groq.py

    # Triage a custom ticket from the command line:
    python agent_groq.py "My booking is stuck in pending and I travel tomorrow."

Dependencies:
    pip install groq

Environment:
    GROQ_API_KEY -- your Groq API key (https://console.groq.com)
"""

# =============================================================================
# Imports
# =============================================================================

import json
import os
import sys
from datetime import datetime
from groq import Groq


# =============================================================================
# Client Initialisation
# =============================================================================

# The Groq client reads GROQ_API_KEY from the environment automatically.
# Set it before running:
#   PowerShell : $env:GROQ_API_KEY="your-key-here"
#   bash/zsh   : export GROQ_API_KEY="your-key-here"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# =============================================================================
# Tool Definitions
# =============================================================================
# Tools follow the OpenAI-compatible function calling schema.
# The model decides which tools to call and in what order.
# Each tool has a name, a description, and a JSON schema for its parameters.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_ticket",
            "description": (
                "Classifies the support ticket by urgency level and category. "
                "This should always be called first to establish priority."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urgency": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                        "description": "Urgency level of the ticket."
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "technical",
                            "billing",
                            "accessibility",
                            "account",
                            "general"
                        ],
                        "description": "Category that best describes the support request."
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "Model confidence in the classification, "
                            "expressed as a float between 0.0 and 1.0."
                        )
                    }
                },
                "required": ["urgency", "category", "confidence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_ticket_info",
            "description": (
                "Extracts key structured fields from the ticket text. "
                "Call after classify_ticket to build the full picture."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_issue": {
                        "type": "string",
                        "description": "A single sentence summarising the core problem."
                    },
                    "affected_service": {
                        "type": "string",
                        "description": "The specific service, feature, or component affected."
                    },
                    "user_sentiment": {
                        "type": "string",
                        "enum": ["frustrated", "neutral", "urgent", "confused"],
                        "description": "The dominant emotional tone detected in the ticket."
                    },
                    "requires_human_review": {
                        "type": "boolean",
                        "description": (
                            "True if the ticket is complex or sensitive enough "
                            "to require a human agent to review before responding."
                        )
                    }
                },
                "required": [
                    "user_issue",
                    "affected_service",
                    "user_sentiment",
                    "requires_human_review"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_ticket",
            "description": (
                "Determines the correct internal team to handle the ticket "
                "and sets an SLA target based on urgency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assigned_team": {
                        "type": "string",
                        "enum": [
                            "engineering",
                            "operations",
                            "billing",
                            "customer_success",
                            "accessibility_team"
                        ],
                        "description": "The team best suited to resolve this ticket."
                    },
                    "sla_hours": {
                        "type": "integer",
                        "description": (
                            "Target first-response time in hours, "
                            "derived from the urgency classification."
                        )
                    },
                    "escalate": {
                        "type": "boolean",
                        "description": (
                            "True if the ticket should be escalated to senior "
                            "staff or management immediately."
                        )
                    }
                },
                "required": ["assigned_team", "sla_hours", "escalate"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_response",
            "description": (
                "Drafts a professional, empathetic first-response email to send "
                "to the user. The response should acknowledge the issue, set "
                "expectations, and outline next steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_line": {
                        "type": "string",
                        "description": "The email subject line for the response."
                    },
                    "response_body": {
                        "type": "string",
                        "description": (
                            "The full email body. Should be empathetic, "
                            "professional, and include clear next steps."
                        )
                    }
                },
                "required": ["subject_line", "response_body"]
            }
        }
    }
]


# =============================================================================
# Tool Handler
# =============================================================================

def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Simulates tool execution and returns a JSON result string.

    In a production deployment, each branch would call a real service:
        classify_ticket     -> internal ML classifier or rules engine
        extract_ticket_info -> CRM lookup or NLP pipeline
        route_ticket        -> ticketing system (e.g. Jira, Zendesk)
        draft_response      -> email dispatch via SendGrid or similar

    For this demonstration, the tool input is echoed back as a success
    response so the agent can continue its reasoning loop.

    Args:
        tool_name  (str):  Name of the tool being called.
        tool_input (dict): Validated arguments provided by the model.

    Returns:
        str: JSON string representing the tool result.
    """
    return json.dumps({"status": "success", "data": tool_input})


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are a support ticket triage agent for Transreport, a UK technology company
that helps transport operators deliver accessible journeys for passengers with
disabilities and accessibility needs.

Your role is to process incoming support tickets systematically by calling the
following tools in order:

    1. classify_ticket      - Determine urgency and category
    2. extract_ticket_info  - Extract structured information from the ticket
    3. route_ticket         - Assign to the correct team with an SLA target
    4. draft_response       - Write a professional first-response email

You must call all four tools for every ticket, regardless of complexity.
Many users contacting Transreport have accessibility requirements -- always
approach responses with empathy appropriate to the situation.
""".strip()


# =============================================================================
# Core Agent Loop
# =============================================================================

def run_triage_agent(ticket_text: str) -> dict:
    """
    Runs the full triage pipeline for a single support ticket.

    Sends the ticket to the LLM and enters an agentic loop, processing tool
    calls until the model signals completion (no further tool calls returned).
    Structured output from each tool is collected into a single result dict.

    Args:
        ticket_text (str): Raw text of the incoming support ticket.

    Returns:
        dict: Structured triage result containing classification, extracted
              info, routing decision, and draft response.
    """
    print("\n" + "=" * 60)
    print("SUPPORT TICKET TRIAGE AGENT")
    print("=" * 60)
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nIncoming Ticket:\n{ticket_text}\n")
    print("-" * 60)
    print("Running agent...\n")

    # Initialise conversation with system context and the raw ticket
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please triage the following support ticket:\n\n{ticket_text}"
        }
    ]

    # Result container -- populated as each tool is called
    triage_result = {
        "ticket_text": ticket_text,
        "timestamp": datetime.now().isoformat(),
        "classification": None,
        "extracted_info": None,
        "routing": None,
        "draft_response": None
    }

    # ------------------------------------------------------------------
    # Agentic loop: continue calling the model until it stops using tools
    # ------------------------------------------------------------------
    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048
        )

        message = response.choices[0].message

        # Append the assistant turn to message history
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        })

        # No tool calls returned -- agent has finished its reasoning
        if not message.tool_calls:
            if message.content:
                print(f"Agent Summary:\n{message.content}")
            break

        # Process each tool call issued in this response
        for tool_call in message.tool_calls:
            tool_name  = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            print(f"  [TOOL] {tool_name}")

            # Execute the tool and capture its result
            result = handle_tool_call(tool_name, tool_input)

            # Store result and log the most relevant fields
            if tool_name == "classify_ticket":
                triage_result["classification"] = tool_input
                print(
                    f"         Urgency: {tool_input.get('urgency')} | "
                    f"Category: {tool_input.get('category')} | "
                    f"Confidence: {tool_input.get('confidence', 0):.0%}"
                )

            elif tool_name == "extract_ticket_info":
                triage_result["extracted_info"] = tool_input
                print(f"         Issue: {tool_input.get('user_issue')}")
                print(f"         Sentiment: {tool_input.get('user_sentiment')}")

            elif tool_name == "route_ticket":
                triage_result["routing"] = tool_input
                print(
                    f"         Team: {tool_input.get('assigned_team')} | "
                    f"SLA: {tool_input.get('sla_hours')}h | "
                    f"Escalate: {tool_input.get('escalate')}"
                )

            elif tool_name == "draft_response":
                triage_result["draft_response"] = tool_input
                print(f"         Subject: {tool_input.get('subject_line')}")
                print("         Draft response prepared.")

            print()

            # Return the tool result to the model so it can continue
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return triage_result


# =============================================================================
# Report Printer
# =============================================================================

def print_triage_report(result: dict) -> None:
    """
    Formats and prints the completed triage result to stdout.

    Args:
        result (dict): The triage result returned by run_triage_agent().
    """
    print("\n" + "=" * 60)
    print("TRIAGE REPORT")
    print("=" * 60)

    if result["classification"]:
        c = result["classification"]
        print("\nCLASSIFICATION")
        print(f"  Urgency    : {c.get('urgency', 'N/A').upper()}")
        print(f"  Category   : {c.get('category', 'N/A')}")
        print(f"  Confidence : {c.get('confidence', 0):.0%}")

    if result["extracted_info"]:
        e = result["extracted_info"]
        print("\nEXTRACTED INFO")
        print(f"  Issue        : {e.get('user_issue', 'N/A')}")
        print(f"  Affected     : {e.get('affected_service', 'N/A')}")
        print(f"  Sentiment    : {e.get('user_sentiment', 'N/A')}")
        print(
            f"  Human review : "
            f"{'Required' if e.get('requires_human_review') else 'Not required'}"
        )

    if result["routing"]:
        r = result["routing"]
        print("\nROUTING")
        print(f"  Team     : {r.get('assigned_team', 'N/A')}")
        print(f"  SLA      : {r.get('sla_hours', 'N/A')} hours")
        print(f"  Escalate : {'Yes' if r.get('escalate') else 'No'}")

    if result["draft_response"]:
        d = result["draft_response"]
        print("\nDRAFT RESPONSE")
        print(f"  Subject : {d.get('subject_line', 'N/A')}")
        print(f"\n{d.get('response_body', 'N/A')}")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# Sample Tickets
# =============================================================================
# Realistic ticket examples representing Transreport's core use cases.
# Used when the script is run without command-line arguments.

SAMPLE_TICKETS = [
    {
        "id": "TKT-001",
        "text": (
            "Hi, I am a wheelchair user and I booked rail assistance for my journey tomorrow "
            "from Manchester Piccadilly to London Euston. The app is showing my booking as "
            "'pending' for the last 3 hours and I cannot confirm if the assistance has been "
            "arranged. I am really worried as I cannot travel without it. "
            "Can someone please help urgently?"
        )
    },
    {
        "id": "TKT-002",
        "text": (
            "Hello, I have been charged twice for my subscription this month. "
            "My account shows two payments of 29.99 GBP on 15th March. "
            "Please refund the duplicate charge. My account email is user@example.com"
        )
    },
    {
        "id": "TKT-003",
        "text": (
            "The API integration we set up last week keeps returning 503 errors "
            "during peak hours between 8am and 10am. This is affecting our ability "
            "to process passenger assistance requests in real time. "
            "This is a production issue impacting multiple operators."
        )
    }
]


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Accept a custom ticket passed as a command-line argument
        ticket_text = " ".join(sys.argv[1:])
        result = run_triage_agent(ticket_text)
        print_triage_report(result)

    else:
        # Default: run the demo using the first sample ticket
        ticket = SAMPLE_TICKETS[0]
        print(f"Running demo with sample ticket: {ticket['id']}\n")

        result = run_triage_agent(ticket["text"])
        print_triage_report(result)

        # Persist the structured output to a JSON file
        output_filename = f"triage_result_{ticket['id']}.json"
        with open(output_filename, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Result saved to: {output_filename}")
