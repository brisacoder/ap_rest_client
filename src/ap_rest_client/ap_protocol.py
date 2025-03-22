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
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
import requests
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages.utils import convert_to_openai_messages, convert_to_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, RequestException, Timeout

from ap_rest_client.logging_config import configure_logging
from ap_rest_client.models.graph_config import RemoteAgentConfig, GraphConfig
from ap_rest_client.models.graph_state import GraphState


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


def default_state(state: GraphState) -> Dict[str, Any]:
    """
    A benign default return for nodes in the graph that do not modify state.

    Returns:
        Dict[str, Any]: Default state with empty messages.
    """
    return {
        "thread_state": {
            "messages": [],
            "extended_info": state.thread_state.extended_info,
        }
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
    if not state.thread_state.messages:
        logger.error("GraphState contains no messages")
        return Command(
            goto="exception_node",
            update={"exception_text": "GraphState contains no messages"},
        )

    # Extract the latest user query
    messages = state.thread_state.messages
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
        "agent_id": config["configurable"]["remote_agent"]["id"],
        "input": {"messages": openai_messages},
        "model": config["configurable"]["remote_agent"]["model"],
        "metadata": config["configurable"]["remote_agent"]["metadata"],
    }

    if state.thread_state.extended_info:
        payload["input"].update(state.thread_state.extended_info)

    # Use a context manager for the session
    with requests.Session() as session:
        try:
            remote_agent_url = config["configurable"]["remote_agent"]["url"]
            rest_timeout = config["configurable"]["rest_timeout"]
            response = session.post(
                remote_agent_url,
                headers=json_headers,
                json=payload,
                timeout=rest_timeout,
            )

            # Raise exception for HTTP errors
            response.raise_for_status()

            # Parse response as JSON
            response_data = response.json()
            # Decode JSON response
            decoded_response = decode_response(response_data)

            logger.info(decoded_response)

            ai_messages: List[Dict[Any, Any]] = decoded_response.get("messages", [])
            base_messages = convert_to_messages(ai_messages)

            # This is tricky. In multi-turn conversation we should only add new messages
            # produced by the remote agent, otherwise we will have duplicates.
            # In this App we will assume remote agent only create a single new message but
            # this is not always true

            return Command(
                goto="end_node",
                update={
                    "thread_state": {
                        "messages": state.thread_state.messages + base_messages,
                        "extended_info": state.thread_state.extended_info,
                    }
                },
            )

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
    return {
        "thread_state": {
            "messages": state.thread_state.messages,
            "extended_info": state.thread_state.extended_info,
        }
    }


def exception_node(state: GraphState) -> Dict[str, Any]:
    """
    Node to handle exceptions during graph execution.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict[str, Any]: Default state.
    """
    logger.info("Exception happen while processing graph: %s", state.exception_msg)
    return default_state(state)


# Build the state graph
def build_graph(
    graph_config: Any = GraphConfig, graph_state: Any = GraphState
) -> CompiledGraph:
    """
    Construct a compiled graph that can be invoked stand-alone or used in Langgraph Studio.

    Returns:
        CompiledGraph: A compiled LangGraph state graph.
    """
    builder = StateGraph(state_schema=graph_state, config_schema=graph_config)
    builder.add_node("node_remote_agent", node_remote_agent)
    builder.add_node("end_node", end_node)
    builder.add_node("exception_node", exception_node)

    builder.add_edge(START, "node_remote_agent")
    builder.add_edge("exception_node", END)
    builder.add_edge("end_node", END)
    return builder.compile(name="ap_local_agent")


def process_graph_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the graph input to ensure it contains the necessary fields.

    Args:
        config (Dict[str, Any]): The input dictionary for the graph.

    Returns:
        Dict[str, Any]: The processed input dictionary with validated fields.

    Raises:
        ValueError: If mandatory fields are missing or empty.
    """
    try:
        processed_config = {}
        processed_config["thread_id"] = config.get("thread_id")
        processed_config["rest_timeout"] = config.get(
            "rest_timeout", 30
        )  # Default to 30 seconds
        remote_agent = config.get("remote_agent", {})
        processed_config["remote_agent"] = {
            "url": remote_agent.get("url"),
            "id": remote_agent.get("id", "remote_agent"),
            "model": remote_agent.get("model", "gpt-4o"),
            "metadata": remote_agent.get("metadata", {}),
        }
        return processed_config

    except (ValueError, KeyError, TypeError) as e:
        logger.error("Error processing graph input: %s", e, exc_info=True)
        raise ValueError(f"Error processing graph input: {e}") from e


def invoke_graph(
    thread_input: Dict[str, Any],
    config: Union[Dict[str, Any], BaseModel],  # Accept both dict and Pydantic model
    graph: CompiledGraph,
) -> Optional[Union[dict[Any, Any], list[dict[Any, Any]]]]:
    """
    Invokes the graph with the given messages and safely extracts the last AI-generated message.

    Args:
        messages (List[Dict[str, Any]]): List of messages in OpenAI format to be processed by the graph.
        graph_config (Union[Dict[str, Any], BaseModel]): Graph configuration containing remote agent details.
        graph (Optional[Any]): An optional langgraph CompiledStateGraph object to use; a default graph
                               will be built if none is provided.

    Returns:
        Optional[Union[dict[Any, Any], list[dict[Any, Any]]]]: The list of all messages returned by the graph.
    """

    # Ensure graph_config is a dictionary
    config_dict = config.model_dump() if isinstance(config, BaseModel) else config

    # processed_config = process_graph_config(config_dict)

    runnable_config: RunnableConfig = {"configurable": config_dict}

    try:

        result = graph.invoke(input=thread_input, config=runnable_config)

        thread_state = result.get("thread_state", {})
        if not thread_state:
            raise ValueError("Graph result does not contain 'thread_state'.")
        messages_list = convert_to_openai_messages(thread_state.get("messages", []))
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


def build_thread_input(
    messages: List[BaseMessage], extended_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Constructs the input dictionary for the graph invocation.

    Args:
        messages (List[Dict[str, Any]]): List of messages in OpenAI format.
        remote_agent_url (str): The URL of the remote agent.
        remote_agent_id (str): The ID of the remote agent.
        metadata (Dict[str, Any]): Metadata for the remote agent.
        opaque_info (Optional[Dict[str, Any]]): Opaque information to be added under input.

    Returns:
        Dict[str, Any]: The constructed input dictionary.
    """
    thread_input: Dict[str, Any] = {"thread_state": {"messages": messages}}
    # Add extended_info if provided
    if extended_info:
        thread_input["thread_state"]["extended_info"] = extended_info

    return thread_input


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
        graph_config = GraphConfig(
            rest_timeout=rest_timeout,
            # thread_id=str(uuid.uuid4()), If you want to use a specific thread_id, uncomment this line
            remote_agent=RemoteAgentConfig(
                url=remote_agent_url,
                id="remote_agent",
                model="gpt-4o",
                metadata={},
            ),
        )

        messages: List[BaseMessage] = [
            HumanMessage(content="Write a story about a cat")
        ]
        # Extended info for the graph
        # This is an example of how to add additional input to the graph
        github = {
            "github": {
                "repo_url": os.getenv("GITHUB_REPO_URL"),
                "github_token": os.getenv("GITHUB_TOKEN"),
                "branch": os.getenv("GITHUB_BRANCH"),
            }
        }
        thread_input = build_thread_input(messages=messages, extended_info=github)

        # Build default graph
        graph = build_graph(GraphConfig, GraphState)
        result = invoke_graph(
            thread_input=thread_input, config=graph_config, graph=graph
        )
        logger.info("graph result: %s", result)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.error("An error occurred in main execution: %s", e, exc_info=True)
    except Exception as e:
        logger.error(
            "An unexpected error occurred in main execution: %s", e, exc_info=True
        )


# Main execution
if __name__ == "__main__":
    main()
