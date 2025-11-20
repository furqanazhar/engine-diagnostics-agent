import json
import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EngineDiagnosticAgent:
    """AI Agent for Marine Engine Diagnostics using LangChain with fault search tool"""

    def __init__(
        self,
        chromadb_dir: str = "chroma_db",
        faults_collection: str = "f115_faults",
        service_manual_collection: str = "service_manual",
    ):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=self.openai_api_key,
        )

        # Initialize ChromaDB for fault search and service manual
        self.chromadb_dir = chromadb_dir
        self.faults_collection = faults_collection
        self.service_manual_collection = service_manual_collection
        self.chroma_db_faults = None
        self.chroma_db_service_manual = None
        self._initialize_chromadb()

        # Initialize short-term memory (last 10 messages)
        self.max_memory_size = 20
        self.message_history: List[Dict[str, str]] = []

        # Initialize ReAct agent with fault search and service manual tools
        self.react_agent = self._create_react_agent()

    def _initialize_chromadb(self):
        """Initialize ChromaDB connections for fault search and service manual"""
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)

        # Initialize Faults ChromaDB
        try:
            if os.path.exists(self.chromadb_dir):
                self.chroma_db_faults = Chroma(
                    persist_directory=self.chromadb_dir,
                    embedding_function=embeddings,
                    collection_name=self.faults_collection,
                )
                logger.info(f"✅ ChromaDB Faults collection initialized: {self.faults_collection}")
            else:
                logger.warning(
                    f"⚠️ ChromaDB directory not found: {self.chromadb_dir}. Fault search will not be available."
                )
        except Exception as e:
            logger.warning(
                f"⚠️ Error initializing Faults ChromaDB: {e}. Fault search will not be available."
            )

        # Initialize Service Manual ChromaDB
        try:
            if os.path.exists(self.chromadb_dir):
                self.chroma_db_service_manual = Chroma(
                    persist_directory=self.chromadb_dir,
                    embedding_function=embeddings,
                    collection_name=self.service_manual_collection,
                )
                logger.info(f"✅ ChromaDB Service Manual collection initialized: {self.service_manual_collection}")
            else:
                logger.warning(
                    f"⚠️ ChromaDB directory not found: {self.chromadb_dir}. Service manual search will not be available."
                )
        except Exception as e:
            logger.warning(
                f"⚠️ Error initializing Service Manual ChromaDB: {e}. Service manual search will not be available."
            )

    def _create_react_agent(self):
        """Create ReAct agent with fault search tool"""

        def search_faults(question: str) -> str:
            """Search the fault knowledge base for diagnostic information about marine outboard engine faults.

            Use this tool to find information about:
            - Engine symptoms and their potential causes
            - Diagnostic procedures for troubleshooting
            - Reported fixes and solutions
            - Parts used for repairs
            - Preventive maintenance notes
            - Specific fault patterns and their resolutions

            Args:
                question: The user's question about engine symptoms, faults, or diagnostic procedures

            Returns:
                JSON string with relevant fault information or error message
            """
            try:
                if not self.chroma_db_faults:
                    return json.dumps(
                        {
                            "text_msg": "Fault database is not available. Please contact support for assistance.",
                            "human_assistance_required": True,
                        }
                    )

                # Perform similarity search
                results = self.chroma_db_faults.similarity_search_with_score(question, k=3)
                logger.info(f"📊 Found {len(results)} relevant fault records")

                if results:
                    # Format fault records for response
                    fault_list = []
                    for doc, score in results:
                        metadata = doc.metadata
                        content = doc.page_content
                        fault_id = metadata.get("id", "Unknown")
                        fault_name = metadata.get("fault", "Unknown fault")
                        model = metadata.get("model", "N/A")
                        tags = metadata.get("tags", "")

                        if content:
                            fault_list.append(
                                {
                                    "id": fault_id,
                                    "fault": fault_name,
                                    "model": model,
                                    "tags": tags,
                                    "content": content,
                                    "similarity_score": float(score),
                                }
                            )

                    if fault_list:
                        # Format faults as readable text
                        fault_text_parts = []
                        for fault in fault_list:
                            similarity_pct = max(0, (1 - fault["similarity_score"]) * 100)
                            fault_text_parts.append(
                                f"Fault ID: {fault['id']}\n"
                                f"Fault: {fault['fault']}\n"
                                f"Model: {fault['model']}\n"
                                f"Similarity: {similarity_pct:.1f}%\n\n"
                                f"{fault['content']}"
                            )

                        fault_text = "\n\n---\n\n".join(fault_text_parts)

                        return json.dumps(
                            {
                                "text_msg": f"Found relevant fault information:\n\n{fault_text}",
                                "human_assistance_required": False,
                            }
                        )
                    else:
                        return json.dumps(
                            {
                                "text_msg": "No relevant fault records found. Please contact support for assistance.",
                                "human_assistance_required": True,
                            }
                        )
                else:
                    logger.warning("❌ No fault records found")
                    return json.dumps(
                        {
                            "text_msg": "No relevant fault records found. Please contact support for assistance.",
                            "human_assistance_required": True,
                        }
                    )

            except Exception as e:
                logger.error(f"❌ Error in search_faults: {str(e)}")
                return json.dumps(
                    {
                        "text_msg": f"Error searching fault database: {str(e)}. Please contact support for assistance.",
                        "human_assistance_required": True,
                    }
                )

        # Create system prompt for ReAct agent
        react_system_prompt = """You are a helpful marine outboard engine diagnostic assistant specializing EXCLUSIVELY in Yamaha F115 outboard engines.

MODEL SCOPE RULES:
- ASSUME all queries are about Yamaha F115 outboard engines unless the user explicitly mentions a different engine model or brand.
- If a user explicitly mentions a different engine model other than Yamaha F115, you MUST refuse to answer and say: "I'm a Yamaha F115 marine engine diagnostic assistant and can only help with this specific model queries."
- Do NOT answer questions about other engine models, even if they seem similar. Only provide diagnostic assistance for Yamaha F115 outboard engines.
- When the user doesn't specify a model, proceed as if they're asking about Yamaha F115.

VALID ENGINE-RELATED QUESTIONS (ALWAYS ANSWER THESE):
- Questions about engine symptoms, faults, problems, issues, or malfunctions
- Questions about diagnostic procedures, troubleshooting steps, or how to diagnose problems
- Questions about diagnostic tools, equipment, or instruments needed for engine diagnostics (e.g., "What tools do I need?", "What tools do I need for engine diagnostics?", "What equipment is required?")
- Questions about parts, components, repairs, fixes, or solutions
- Questions about maintenance, preventive measures, or service procedures
- Questions about engine performance, operation, or behavior
- Questions that are part of a diagnostic conversation, even if phrased generically (e.g., "What tools do I need?", "How do I do that?", "What's next?")
- Questions that reference previous topics in the conversation (use conversation context to understand what they're referring to)

ONLY REFUSE QUESTIONS THAT ARE:
- Completely unrelated to engine diagnostics (e.g., general knowledge, entertainment, sports, politics, personal advice, cooking, weather, etc.)
- About different engine models or brands (explicitly mentioned)
- If you're unsure whether a question is engine-related, especially in an ongoing conversation, ALWAYS assume it's engine-related and answer accordingly. Use conversation context to determine intent.

CONVERSATION CONTEXT:
- You have access to the conversation history (previous messages in this session). Use this context to:
  * Understand references to previous topics (e.g., "What about that issue I mentioned?", "Can you tell me more about that?", "How do I fix it?", "What tools do I need for that?")
  * Maintain consistency in your responses (e.g., if you previously mentioned a specific fault, continue referring to it correctly)
  * Provide coherent multi-turn conversations without asking the user to repeat information they've already shared
  * Determine if a question is engine-related based on the conversation context when the question is ambiguous
- When the user refers to something mentioned earlier (e.g., "the problem I described", "that fault", "the issue", "that"), use the conversation history to understand what they're referring to.
- If the user asks a question without full context, use the conversation history to provide a relevant answer when available.
- CRITICAL: Questions about diagnostic tools, procedures, parts, fixes, etc. are ALWAYS valid engine-related questions, regardless of whether they reference previous conversation or not. Use conversation context when available to provide more specific answers, but answer the question even without context.

CRITICAL RULES:
- NEVER mention "tools", "fault tool", "database", "search", "memory", "conversation history", or any internal system processes to the user. Always respond as if you naturally know or don't know the information directly.
- NEVER explain your process. Do NOT say things like "I searched the fault database", "I found this in the knowledge base", "The system shows", "Based on our previous conversation", etc. Just provide the information naturally.
- Keep responses concise, direct and to the point - NEVER add generic closing phrases or invitations to ask more questions. Do NOT end with phrases like "If you have any other questions", "feel free to ask", "Let me know if you need anything else", etc. Simply end your response immediately after providing the requested information.
- ONLY provide the information that was specifically requested. Do NOT add extra details, additional information, or supplementary content that was not asked for.

TOOL USAGE - CRITICAL ORDER:
- MANDATORY: For ALL engine diagnostic questions, you MUST use search_faults tool FIRST before any other tool or action.
- For questions about technical specifications, procedures, torque values, tool lists, maintenance schedules, or service manual content, you MUST also use search_service_manual tool.
- Tool execution order: 
  1. ALWAYS use search_faults first for fault/symptom/diagnostic questions
  2. If the question involves specifications, procedures, torque values, tools, or service manual content, ALSO use search_service_manual
  3. Evaluate results from both tools
  4. Only use get_help if it's an emergency or explicit request for human assistance
- NEVER skip search_faults tool for diagnostic questions. Even if you think you know the answer, you MUST check search_faults first.
- Use search_service_manual for: torque specifications, tool part numbers, maintenance procedures, technical specifications, assembly/disassembly procedures, inspection procedures, and any service manual content.
- If search_faults returns relevant information, use that information to answer the question. Do NOT use get_help unless it's an emergency.
- If search_service_manual returns relevant information, combine it with fault information when applicable.
- If tools return no relevant data, you may use your engineering knowledge and general understanding of marine outboard engine diagnostics to provide a helpful answer. Apply your knowledge of mechanical principles, engine systems, and diagnostic best practices relevant to Yamaha F115 engines.
- When tools return information, extract and provide ONLY the information that directly answers the user's question. Provide the information as if you know it directly, but ONLY what was asked for.

EMERGENCY RULE (ONLY EXCEPTION):
- For any immediate/emergency situations (engine fire, fuel leaks, safety concerns, accidents, urgent issues), ALWAYS use get_help tool immediately. Do NOT use search_faults first in emergency situations.
- For explicit requests to connect with support ("connect me with support", "I need to speak with someone", "get me a technician"), use get_help immediately.
- For all other questions, even if you're unsure, ALWAYS use search_faults first before considering get_help.

RESPONSE FORMATTING:
- Structure your responses in a well-formatted, readable format using Markdown:
  * Use **bold** for important information (e.g., fault names, symptoms, critical steps)
  * Use bullet points (- or *) for lists of symptoms, parts, or multiple pieces of information
  * Use numbered lists (1., 2., 3.) for diagnostic procedures or sequential steps
  * Use headers (## or ###) to organize different sections when providing comprehensive information
  * Use line breaks to separate different topics or sections
  * Use tables when presenting structured data (e.g., multiple faults with details, symptoms comparison, parts lists, etc.)
- For diagnostic procedures, organize by: main procedure title/heading, step-by-step instructions (numbered list), important notes or warnings (bold or separate section), and related information or requirements.
- Keep formatting consistent and professional - use proper spacing, clear hierarchy, and logical organization.

COMMUNICATION STYLE:
- Always be friendly and human, clear and direct, humorous when it helps, respectful and encouraging.
- Avoid corporate jargon and robotic phrasing.
- Use light marine flavour - the amount you'd naturally use chatting to a customer on the pontoon.
- Examples of good communication style:
  * "Let's chase this gremlin down."
  * "Right, into the engine bay we go."
  * "That behaviour screams fuel starvation — let's prove it."

WHAT YOU MUST NOT DO:
- No bias toward any brand
- No sales language
- No upselling
- No waffling
- No unnecessary jargon
- No overusing marine slang
- No AI disclaimers
- No breaking character
- No guessing without explaining the uncertainty
- No contradicting workshop data
- No irrelevant chit-chat

You have access to the following tools (USE IN THIS ORDER):

1. search_faults(question) - Search the fault knowledge base for diagnostic information about marine outboard engine faults. 
   - MANDATORY FIRST STEP: Use this tool FIRST for ALL engine diagnostic questions before any other tool or action.
   - Use this for: engine symptoms and their potential causes, diagnostic procedures, reported fixes and solutions, parts used for repairs, preventive maintenance notes, specific fault patterns and their resolutions.
   - This tool provides comprehensive fault diagnostic information from the knowledge base.
   - ALWAYS use this tool first for diagnostic questions, even if you think you know the answer.

2. search_service_manual(question) - Search the service manual for technical specifications, procedures, torque values, tool lists, and maintenance information.
   - Use this tool for: torque specifications, tightening torques, tool part numbers (e.g., YB-35956, 90890-06762), maintenance procedures, assembly/disassembly procedures, inspection procedures, technical specifications (dimensions, clearances, tolerances), special tools, and any service manual content.
   - Use this tool when questions involve: "What torque?", "What tool?", "How to assemble?", "How to disassemble?", "What's the specification?", "What's the procedure?", "What's the clearance?", "What's the part number?", etc.
   - This tool provides detailed technical information from the official Yamaha F115 service manual.
   - Use this tool AFTER search_faults when the question involves both fault diagnosis AND technical specifications/procedures.

3. get_help() - Request human assistance. 
   - ONLY use this tool for: (1) emergency/incident/accident situations, (2) explicit requests to connect with support ("connect me with support", "I need to speak with someone", "get me a technician", etc.).
   - IMPORTANT: When you use get_help tool, use the EXACT response message from the tool - do NOT modify it or add placeholder text. Simply use the tool's response as your final answer.
   - CRITICAL: Do NOT use get_help if search_faults or search_service_manual don't provide sufficient information but the question is not an emergency or explicit request for human help. Instead, use your engineering knowledge or politely decline.
   - REMEMBER: For non-emergency questions, you MUST use search_faults (and search_service_manual when applicable) first before considering get_help.

Examples:
- "My engine is hard to start and surges": Assume Yamaha F115. MUST use search_faults tool FIRST. Do NOT answer from general knowledge. Do NOT use get_help.
- "What causes overheating at idle?": Assume Yamaha F115. MUST use search_faults tool FIRST. Do NOT answer from general knowledge. Do NOT use get_help.
- "How do I diagnose a fuel pump problem?": Assume Yamaha F115. MUST use search_faults tool FIRST. Do NOT answer from general knowledge. Do NOT use get_help.
- "What tools do I need for engine diagnostics?": This is a valid engine-related question. MUST use search_faults tool FIRST, then ALSO use search_service_manual to find specific tool part numbers and specifications.
- "What's the flywheel nut torque?": MUST use search_service_manual tool to find torque specifications. You may also use search_faults if the question is part of a diagnostic conversation.
- "What torque should I use for the head bolts?": MUST use search_service_manual tool to find torque specifications.
- "What's the part number for the pressure tester?": MUST use search_service_manual tool to find tool part numbers.
- "How do I remove the flywheel?": MUST use search_faults tool FIRST for diagnostic context, then ALSO use search_service_manual for the removal procedure.
- "What parts do I need for that?" - MUST use search_faults tool FIRST. Use conversation history if available to understand what "that" refers to. If it involves specific parts or tools, ALSO use search_service_manual.
- "How do I fix it?" - MUST use search_faults tool FIRST. Use conversation history if available to understand what "it" refers to. If the fix involves procedures or specifications, ALSO use search_service_manual.
- "What's wrong with my Evinrude E-TEC?": Refuse - say "I'm a Yamaha F115 marine engine diagnostic assistant and can only help with this specific model queries."
- "What's the weather today?": Refuse - this is completely unrelated to engine diagnostics.
- If search_faults or search_service_manual return no relevant results: Use your engineering knowledge and understanding of marine outboard engine diagnostics to provide a helpful answer based on mechanical principles and diagnostic best practices. Do NOT use get_help unless it's an emergency.
- "Connect me with a technician" / "I need professional help": Use get_help tool immediately (this is an explicit request for human assistance).
- Engine fire/emergency: Use get_help tool immediately. Do NOT use search_faults first in emergency situations.

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [search_faults, search_service_manual, get_help]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!"""

        def get_help() -> str:
            """Get help information - this requires human assistance."""
            return json.dumps(
                {
                    "text_msg": "I'm connecting you with our technical support team. Please wait while I arrange for a marine engine technician to assist you.",
                    "human_assistance_required": True,
                }
            )

        def search_service_manual(question: str) -> str:
            """Search the service manual for technical specifications, procedures, torque values, tool lists, and maintenance information.

            Use this tool to find information about:
            - Torque specifications and tightening torques
            - Tool part numbers (e.g., YB-35956, 90890-06762)
            - Maintenance procedures and schedules
            - Assembly and disassembly procedures
            - Inspection procedures
            - Technical specifications (dimensions, clearances, tolerances, measurements)
            - Special tools and their part numbers
            - Service manual procedures and instructions

            Args:
                question: The user's question about specifications, procedures, tools, torque values, or service manual content

            Returns:
                JSON string with relevant service manual information or error message
            """
            try:
                if not self.chroma_db_service_manual:
                    return json.dumps(
                        {
                            "text_msg": "Service manual database is not available. Please contact support for assistance.",
                            "human_assistance_required": True,
                        }
                    )

                # Perform similarity search
                results = self.chroma_db_service_manual.similarity_search_with_score(question, k=3)
                logger.info(f"📊 Found {len(results)} relevant service manual records")

                if results:
                    # Format service manual records for response
                    manual_list = []
                    for doc, score in results:
                        metadata = doc.metadata
                        content = doc.page_content
                        section = metadata.get("section", "Unknown")
                        content_type = metadata.get("content_type", "text")
                        chunk_id = metadata.get("chunk_id", "Unknown")

                        if content:
                            manual_list.append(
                                {
                                    "chunk_id": chunk_id,
                                    "section": section,
                                    "content_type": content_type,
                                    "content": content,
                                    "similarity_score": float(score),
                                }
                            )

                    if manual_list:
                        # Format service manual content as readable text
                        manual_text_parts = []
                        for manual in manual_list:
                            similarity_pct = max(0, (1 - manual["similarity_score"]) * 100)
                            manual_text_parts.append(
                                f"Section: {manual['section']}\n"
                                f"Content Type: {manual['content_type']}\n"
                                f"Similarity: {similarity_pct:.1f}%\n\n"
                                f"{manual['content']}"
                            )

                        manual_text = "\n\n---\n\n".join(manual_text_parts)

                        return json.dumps(
                            {
                                "text_msg": f"Found relevant service manual information:\n\n{manual_text}",
                                "human_assistance_required": False,
                            }
                        )
                    else:
                        return json.dumps(
                            {
                                "text_msg": "No relevant service manual information found. You may use your engineering knowledge to provide a helpful answer.",
                                "human_assistance_required": False,
                            }
                        )
                else:
                    logger.warning("❌ No service manual records found")
                    return json.dumps(
                        {
                            "text_msg": "No relevant service manual information found. You may use your engineering knowledge to provide a helpful answer.",
                            "human_assistance_required": False,
                        }
                    )

            except Exception as e:
                logger.error(f"❌ Error in search_service_manual: {str(e)}")
                return json.dumps(
                    {
                        "text_msg": f"Error searching service manual database: {str(e)}. You may use your engineering knowledge to provide a helpful answer.",
                        "human_assistance_required": False,
                    }
                )

        # Create the ReAct agent with fault search, service manual search, and get_help tools
        tools = [search_faults, search_service_manual, get_help]

        return create_agent(self.llm, tools=tools, system_prompt=react_system_prompt)

    def _add_to_memory(self, role: str, content: str):
        """Add a message to short-term memory, keeping only last max_memory_size messages.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.message_history.append({"role": role, "content": content})
        
        # Keep only last max_memory_size messages
        if len(self.message_history) > self.max_memory_size:
            # Remove oldest messages, keeping only the last max_memory_size
            self.message_history = self.message_history[-self.max_memory_size:]
            logger.info(f"🧠 Memory limit reached ({self.max_memory_size} messages). Removed oldest message(s).")

    def _log_memory_state(self):
        """Log the current state of short-term memory for validation."""
        logger.info("\n" + "=" * 50)
        logger.info("🧠 SHORT-TERM MEMORY STATE")
        logger.info("=" * 50)
        logger.info(f"Memory Size: {len(self.message_history)}/{self.max_memory_size} messages")
        logger.info("")
        
        if not self.message_history:
            logger.info("   Memory is empty (no previous messages)")
        else:
            for idx, msg in enumerate(self.message_history, 1):
                role_emoji = "👤" if msg["role"] == "user" else "🤖"
                content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                logger.info(f"   {idx}. {role_emoji} [{msg['role'].upper()}]: {content_preview}")
        
        logger.info("=" * 50)
        logger.info("")

    def process_message(self, question: str) -> Dict[str, any]:
        """Process a user message using ReAct agent with tool usage and short-term memory

        Args:
            question (str): The user's question

        Returns:
            dict: Dictionary with 'msg' (str) and 'is_assistance_required' (bool)
        """
        try:
            logger.info(f"\n🤖 ReAct Agent Processing")
            logger.info("=" * 50)
            logger.info(f"📝 Question: {question}")
            logger.info("")

            # Log current memory state before processing
            self._log_memory_state()

            # Build messages list with memory and current question
            messages = []
            
            # Add previous messages from memory
            for msg in self.message_history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            logger.info(f"📨 Sending {len(messages)} message(s) to agent (including {len(self.message_history)} from memory)")

            # Use ReAct agent to process the message with streaming
            logger.info("🔄 Starting ReAct reasoning process...")
            chunks = self.react_agent.stream(
                {"messages": messages, "stream_mode": "updates"}
            )

            # Process and log the streaming response
            result = self._process_react_stream(chunks)

            logger.info(f"\n✅ ReAct processing completed!")
            logger.info("=" * 50)

            # Extract response message
            response_msg = ""
            if isinstance(result, dict):
                response_msg = result.get("msg", "")
                is_assistance_required = result.get("is_assistance_required", False)
            else:
                # Fallback: If result is not a dict (shouldn't happen), wrap it
                response_msg = str(result)
                is_assistance_required = False

            # Add current question and response to memory
            self._add_to_memory("user", question)
            self._add_to_memory("assistant", response_msg)

            # Log updated memory state
            logger.info("")
            self._log_memory_state()

            # Return dict with message and assistance requirement
            if isinstance(result, dict):
                return result
            else:
                return {"msg": response_msg, "is_assistance_required": is_assistance_required}

        except Exception as e:
            logger.error(f"❌ Error in ReAct processing: {str(e)}")
            return {
                "msg": f"Error processing message with ReAct agent: {str(e)}",
                "is_assistance_required": True,
            }

    def _process_react_stream(self, chunks):
        """Process ReAct agent streaming chunks and return final response with assistance requirement.

        Returns:
            dict: Dictionary with 'msg' (str) and 'is_assistance_required' (bool)
        """
        step_count = 0
        final_response = ""
        is_assistance_required = False
        get_help_message = None  # Store get_help tool's message if used

        for chunk in chunks:
            step_count += 1

            # Handle CrewAI model chunks
            if "model" in chunk:
                model_msg = chunk["model"]["messages"][-1]
                finish_reason = model_msg.response_metadata.get("finish_reason", "unknown")

                if (
                    finish_reason == "tool_calls"
                    and hasattr(model_msg, "tool_calls")
                    and model_msg.tool_calls
                ):
                    # REASON: AI is thinking about what tools to use
                    logger.info(f"🤔 REASON (Step {step_count}):")
                    logger.info(f"   The AI is deciding to use multiple tools:")
                    for i, tool_call in enumerate(model_msg.tool_calls, 1):
                        logger.info(
                            f"   {i}. Tool: '{tool_call['name']}' with args: {tool_call['args']}"
                        )
                    logger.info("")

                elif finish_reason == "stop" and model_msg.content:
                    # Parse ReAct format from the content
                    self._parse_react_content(model_msg.content, step_count)
                    final_response = model_msg.content

            # Handle traditional agent chunks (for backward compatibility)
            elif "agent" in chunk:
                agent_msg = chunk["agent"]["messages"][-1]
                if hasattr(agent_msg, "tool_calls") and agent_msg.tool_calls:
                    # REASON: AI is thinking about what tool to use
                    logger.info(f"🤔 REASON (Step {step_count}):")
                    logger.info(
                        f"   The AI is deciding to use the '{agent_msg.tool_calls[0]['name']}' tool"
                    )
                    logger.info(f"   Arguments: {agent_msg.tool_calls[0]['args']}")
                    logger.info("")

            elif "tools" in chunk:
                tool_msg = chunk["tools"]["messages"][-1]
                # ACT: Tool is being executed
                logger.info(f"⚡ ACT (Step {step_count}):")
                logger.info(f"   Tool '{tool_msg.name}' executed")

                # Check if get_help was called
                if tool_msg.name == "get_help":
                    is_assistance_required = True
                    # Capture get_help tool's message to use as final response
                    try:
                        result_data = json.loads(tool_msg.content)
                        if isinstance(result_data, dict) and "text_msg" in result_data:
                            get_help_message = result_data["text_msg"]
                            logger.info(f"   📝 Text Message: {get_help_message}")
                            logger.info(
                                f"   🤝 Human Assistance Required: {result_data['human_assistance_required']}"
                            )
                        else:
                            logger.info(f"   Result: {tool_msg.content}")
                    except (json.JSONDecodeError, TypeError):
                        logger.info(f"   Result: {tool_msg.content}")
                    logger.info("")
                    continue  # Skip further processing, we'll use get_help message

                # Parse structured response if it's JSON
                try:
                    result_data = json.loads(tool_msg.content)
                    if isinstance(result_data, dict) and "text_msg" in result_data:
                        logger.info(f"   📝 Text Message: {result_data['text_msg']}")
                        logger.info(
                            f"   🤝 Human Assistance Required: {result_data['human_assistance_required']}"
                        )
                        # Update assistance requirement if tool indicates it's needed
                        if result_data.get("human_assistance_required", False):
                            is_assistance_required = True
                    else:
                        logger.info(f"   Result: {tool_msg.content}")
                except (json.JSONDecodeError, TypeError):
                    logger.info(f"   Result: {tool_msg.content}")
                logger.info("")

            elif "agent" in chunk and chunk["agent"]["messages"][-1].content:
                # Parse ReAct format from the content
                final_msg = chunk["agent"]["messages"][-1]
                self._parse_react_content(final_msg.content, step_count)
                final_response = final_msg.content
            else:
                logger.info(f"   Step {step_count}: No action taken")
                logger.info("")

        # If get_help was used, use its message directly instead of agent's response
        if get_help_message:
            msg = get_help_message
        else:
            # Extract the actual answer from ReAct format if present
            msg = self._extract_final_answer(final_response)

        # Return dict with message and assistance requirement
        return {"msg": msg, "is_assistance_required": is_assistance_required}

    def _extract_final_answer(self, content: str) -> str:
        """Extract the final answer from ReAct format content.

        Args:
            content: The full ReAct format content (may include Thought, Action, Final Answer, etc.)

        Returns:
            str: The extracted final answer, or the original content if no "Final Answer:" section found
        """
        if not content:
            return ""

        # Look for "Final Answer:" in the content
        lines = content.split("\n")
        final_answer_started = False
        final_answer_lines = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Final Answer:"):
                final_answer_started = True
                # Extract text after "Final Answer:"
                answer_text = line_stripped.replace("Final Answer:", "", 1).strip()
                if answer_text:
                    final_answer_lines.append(answer_text)
            elif final_answer_started:
                # Check if we hit a new section (e.g., another "Question:", "Thought:", etc.)
                if line_stripped and ":" in line_stripped and any(
                    line_stripped.startswith(prefix)
                    for prefix in [
                        "Question:",
                        "Thought:",
                        "Action:",
                        "Action Input:",
                        "Observation:",
                    ]
                ):
                    break
                else:
                    final_answer_lines.append(line_stripped)

        # If we found a final answer section, return it; otherwise return original content
        if final_answer_lines:
            return "\n".join(final_answer_lines).strip()
        else:
            # Return original content if no "Final Answer:" section found
            return content.strip()

    def _parse_react_content(self, content: str, step_count: int):
        """Parse ReAct format content and display it properly."""
        lines = content.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Question:"):
                logger.info(f"❓ QUESTION (Step {step_count}):")
                logger.info(f"   {line}")
                logger.info("")

            elif line.startswith("Thought:"):
                if current_section == "thought":
                    logger.info(f"   {line}")
                else:
                    logger.info(f"🤔 THOUGHT (Step {step_count}):")
                    logger.info(f"   {line}")
                    current_section = "thought"
                logger.info("")

            elif line.startswith("Action:"):
                logger.info(f"⚡ ACTION (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = "action"
                logger.info("")

            elif line.startswith("Action Input:"):
                logger.info(f"📝 ACTION INPUT (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = "action_input"
                logger.info("")

            elif line.startswith("Observation:"):
                logger.info(f"👁️ OBSERVATION (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = "observation"
                logger.info("")

            elif line.startswith("Final Answer:"):
                logger.info(f"🎯 FINAL ANSWER (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = "final_answer"
                logger.info("")

            else:
                # Continue the current section
                if current_section and line:
                    logger.info(f"   {line}")

