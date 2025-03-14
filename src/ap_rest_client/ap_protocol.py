"""
This module defines the protocol for interacting with a remote agent service via REST API.
It includes functions for loading environment variables, decoding responses, and handling
graph state and configuration. The main functionality revolves around building and invoking
a state graph that communicates with the remote agent service to process messages and handle
exceptions.
"""

import json
import logging
import os
import traceback
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import requests
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, RequestException, Timeout

from ap_rest_client.logging_config import configure_logging

# Step 1: Initialize a basic logger first (to avoid errors before full configuration)
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Minimal level before full configuration
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def load_environment_variables(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file safely.

    Args:
        env_file (Optional[str]): Path to a specific `.env` file. If None,
                                  it searches for a `.env` file automatically.

    Returns:
        None
    """
    env_path = env_file or find_dotenv()

    if env_path:
        load_dotenv(env_path, override=True)
        logger.info(".env file loaded from %s", env_path)
    else:
        logger.warning("No .env file found. Ensure environment variables are set.")


def decode_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decodes the JSON response from the remote server and extracts relevant information.

    Args:
        response_data (Dict[str, Any]): The JSON response from the server.

    Returns:
        Dict[str, Any]: A structured dictionary containing extracted response fields.
    """
    try:
        agent_id = response_data.get("agent_id", "Unknown")
        output = response_data.get("output", {})
        model = response_data.get("model", "Unknown")
        metadata = response_data.get("metadata", {})

        # Extract messages if present
        messages = output.get("messages", [])

        return {
            "agent_id": agent_id,
            "messages": messages,
            "model": model,
            "metadata": metadata,
        }
    except (ValueError, TypeError, KeyError) as e:
        return {"error": f"Failed to decode response: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


class GraphState(BaseModel):
    """Represents the state of the graph, containing messages and an optional error message."""

    messages: Annotated[List[BaseMessage], add_messages] = Field(
        ..., description="List of messages exchanged within the graph session."
    )

    exception_msg: Optional[str] = Field(
        None,
        description="Optional error message in case of exceptions during execution.",
    )


class GraphConfig(BaseModel):
    """Configuration for the graph execution, including remote agent details."""

    remote_agent_url: str = Field(
        "http://127.0.0.1:8123/api/v1/runs",
        description="URL of the remote agent service. Defaults to local server.",
    )

    thread_id: Optional[str] = Field(
        None, description="Optional unique identifier for the execution thread."
    )

    rest_timeout: int = Field(
        30, description="Timeout (in seconds) for REST API requests."
    )


def default_state() -> Dict[str, Any]:
    """
    A benign default return for nodes in the graph that do not modify state.

    Returns:
        Dict[str, Any]: Default state with empty messages.
    """
    return {
        "messages": [],
    }


# Graph node that makes a stateless request to the Remote Graph Server
def node_remote_agent(
    state: GraphState, config: RunnableConfig
) -> Command[Literal["exception_node", "end_node"]]:
    """
    Sends a stateless request to the Remote Graph Server.

    Args:
        state (GraphState): The current graph state containing messages.
        config (RunnableConfig): Configuration for the graph execution.

    Returns:
        Command[Literal["exception_node", "end_node"]]: Command to transition to the next node.
    """
    if not state.messages:
        logger.error("GraphState contains no messages")
        return Command(
            goto="exception_node",
            update={"exception_text": "GraphState contains no messages"},
        )

    # Extract the latest user query
    messages = state.messages
    human_message = messages[-1].content
    logger.info("sending message: %s", human_message)

    # Request headers
    json_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    openai_messages = convert_to_openai_messages(messages)

    # payload to send to autogen server at /runs endpoint
    payload = {
        "agent_id": "remote_agent",
        "input": {"messages": openai_messages},
        "model": "gpt-4o",
        "metadata": {"id": str(uuid.uuid4())},
    }

    # Use a context manager for the session
    with requests.Session() as session:
        try:
            remote_agent_url = config["configurable"].get("remote_agent_url")
            rest_timeout = config["configurable"].get("rest_timeout")
            response = session.post(
                remote_agent_url, headers=json_headers, json=payload, timeout=rest_timeout
            )

            # Raise exception for HTTP errors
            response.raise_for_status()

            # Parse response as JSON
            response_data = response.json()
            # Decode JSON response
            decoded_response = decode_response(response_data)

            logger.info(decoded_response)

            messages = decoded_response.get("messages", [])

            # This is tricky. In multi-turn conversation we should only add new messages
            # produced by the remote agent, otherwise we will have duplicates.
            # In this App we will assume remote agent only create a single new message but
            # this is not always true

            return Command(goto="end_node", update={"messages": messages[-1]})

        except Timeout as timeout_err:
            error_msg = {
                "error": "Connection timeout",
                "exception": str(timeout_err),
            }
            logger.error(json.dumps(error_msg))
            return Command(
                goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
            )

        except RequestsConnectionError as conn_err:
            error_msg = {
                "error": "Connection failure",
                "exception": str(conn_err),
            }
            logger.error(json.dumps(error_msg))
            return Command(
                goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
            )

        except HTTPError as http_err:
            error_msg = {
                "error": "HTTP request failed",
                "status_code": response.status_code,
                "exception": str(http_err),
            }
            logger.error(json.dumps(error_msg))
            return Command(
                goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
            )

        except RequestException as req_err:
            error_msg = {"error": "Request failed", "exception": str(req_err)}
            logger.error(json.dumps(error_msg))
            return Command(
                goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
            )

        except json.JSONDecodeError as json_err:
            error_msg = {"error": "Invalid JSON response", "exception": str(json_err)}
            logger.error(json.dumps(error_msg))
            return Command(
                goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
            )

        except (ValueError, TypeError, KeyError) as e:
            error_msg = {
                "error": "Unexpected failure",
                "exception": str(e),
                "stack_trace": traceback.format_exc(),
            }
            logger.error(json.dumps(error_msg))
            return Command(
                goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
            )

        except Exception as e:
            error_msg = {
                "error": "An unexpected error occurred",
                "exception": str(e),
                "stack_trace": traceback.format_exc(),
            }
            logger.error(json.dumps(error_msg))
            return Command(
                goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
            )


# Graph node that makes a stateless request to the Remote Graph Server
def end_node(state: GraphState) -> Dict[str, Any]:
    """
    Final node in the graph execution.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict[str, Any]: Default state.
    """
    logger.info("Thread end: %s", state.model_dump().values())
    return default_state()


def exception_node(state: GraphState) -> Dict[str, Any]:
    """
    Node to handle exceptions during graph execution.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict[str, Any]: Default state.
    """
    logger.info("Exception happen while processing graph: %s", state.exception_msg)
    return default_state()


# Build the state graph
def build_graph() -> CompiledGraph:
    """
    Construct a compiled graph that can be invoked stand-alone or used in Langgraph Studio.

    Returns:
        CompiledGraph: A compiled LangGraph state graph.
    """
    builder = StateGraph(state_schema=GraphState, config_schema=GraphConfig)
    builder.add_node("node_remote_agent", node_remote_agent)
    builder.add_node("end_node", end_node)
    builder.add_node("exception_node", exception_node)

    builder.add_edge(START, "node_remote_agent")
    builder.add_edge("exception_node", END)
    builder.add_edge("end_node", END)
    return builder.compile(name="ap_local_agent")


def invoke_graph(
    messages: List[Dict[str, str]],
    graph: Optional[Any] = None,
    remote_agent_url: Optional[str] = None,
    rest_timeout: Optional[int] = None,
) -> Optional[Union[dict[Any, Any], list[dict[Any, Any]]]]:
    """
    Invokes the graph with the given messages and safely extracts the last AI-generated message.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries in OpenAI format.
        graph (Optional[Any]): An optional langgraph CompiledStateGraph object to use;
        default graph will be built if none provided.
        remote_agent_url (Optional[str]): The URL for the remote agent.
        rest_timeout (Optional[int]): The timeout for REST requests.

    Returns:
        Optional[Union[
            dict[Any, Any],
            list[dict[Any, Any]]
        ]]: The list of all messages returned by the graph.
    """

    load_environment_variables()

    # Apply precedence rules for remote_agent_url
    if remote_agent_url is None:
        remote_agent_url = os.getenv(
            "REMOTE_AGENT_URL", "http://127.0.0.1:8123/api/v1/runs"
        )

    # Same for rest_timeout
    if rest_timeout is None:
        rest_timeout = int(os.getenv("rest_timeout", "30"))

    inputs = {"messages": messages}
    logger.debug({"event": "invoking_graph", "inputs": inputs})

    graph_config = GraphConfig(
        remote_agent_url=remote_agent_url,
        rest_timeout=rest_timeout,
        thread_id=str(uuid.uuid4()),
    )

    config: RunnableConfig = {"configurable": graph_config.model_dump()}

    try:
        if not graph:
            graph = build_graph()

        result = graph.invoke(inputs, config=config)

        if not isinstance(result, dict):
            raise TypeError(
                f"Graph invocation returned non-dict result: {type(result)}"
            )

        messages_list = convert_to_openai_messages(result.get("messages", []))
        if not isinstance(messages_list, list) or not messages_list:
            raise ValueError("Graph result does not contain a valid 'messages' list.")

        last_message = messages_list[-1]
        if not isinstance(last_message, dict) or "content" not in last_message:
            raise KeyError(f"Last message does not contain 'content': {last_message}")

        ai_message_content = last_message["content"]
        logger.info("AI message content: %s", ai_message_content)
        return messages_list

    except (TypeError, ValueError, KeyError) as e:
        logger.error("Error invoking graph: %s", e, exc_info=True)
        return [{"role": "assistant", "content": "Error processing user message"}]
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        return [{"role": "assistant", "content": "An unexpected error occurred"}]


def main() -> None:
    """
    Main function to load environment variables, configure logging, and invoke the graph.
    """
    load_environment_variables()
    _ = configure_logging()

    remote_agent_url = os.getenv(
        "REMOTE_AGENT_URL", "http://127.0.0.1:8123/api/v1/runs"
    )
    rest_timeout = int(os.getenv("REST_TIMEOUT", "30"))
    try:
        graph = build_graph()
        graph_config = GraphConfig(
            remote_agent_url=remote_agent_url,
            rest_timeout=rest_timeout,
            thread_id=str(uuid.uuid4()),
        )
        config: RunnableConfig = {"configurable": graph_config.model_dump()}
        inputs = {"messages": [HumanMessage(content="Write a story about a cat")]}
        logger.info({"event": "invoking_graph", "inputs": inputs})
        result = graph.invoke(inputs, config=config)
        logger.info({"event": "final_result", "result": result})
    except Exception as e:
        logger.error(
            "An unexpected error occurred in main execution: %s", e, exc_info=True)


# Main execution
if __name__ == "__main__":
    main()
