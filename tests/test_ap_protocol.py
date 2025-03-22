"""
Unit tests for the `node_remote_agent` function in the `ap_protocol` module.
These tests cover both successful and failure scenarios for the function.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from ap_rest_client.ap_protocol import (
    GraphConfig,
    GraphState,
    build_graph,
    build_thread_input,
    invoke_graph,
)
from ap_rest_client.models.graph_config import RemoteAgentConfig


class TestNodeRemoteAgent(unittest.TestCase):
    """
    Test suite for the `node_remote_agent` function.
    """

    @patch("requests.Session.post")
    def test_node_remote_agent_success(self, mock_post):
        """
        Test the `node_remote_agent` function for a successful response.

        This test mocks a successful HTTP POST request and verifies that
        the function processes the response correctly.
        """
        # Mock the response from session.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agent_id": "test_agent",
            "output": {"messages": [{"role": "assistant", "content": "Hello!"}]},
            "model": "gpt-4o",
            "metadata": {},
        }
        mock_post.return_value = mock_response

        # Prepare test inputs
        messages = [HumanMessage(content="Hi")]
        github = {
            "github": {
                "repo_url": os.getenv("GITHUB_REPO_URL"),
                "github_token": os.getenv("GITHUB_TOKEN"),
                "branch": os.getenv("GITHUB_BRANCH"),
            }
        }
        thread_input = build_thread_input(messages=messages, extended_info=github)
        graph_config = GraphConfig(
            rest_timeout=30,
            # thread_id=str(uuid.uuid4()), If you want to use a specific thread_id, uncomment this line
            remote_agent=RemoteAgentConfig(
                url="http://example.com",
                id="remote_agent",
                model="gpt-4o",
                metadata={},
            ),
        )
        graph = build_graph(GraphConfig, GraphState)

        # Call the graph
        result = invoke_graph(
            thread_input=thread_input, config=graph_config, graph=graph
        )

        # Assertions
        self.assertEqual(result, "end_node")
        self.assertIn("messages", result.update)
        self.assertEqual(result.update["messages"]["content"], "Hello!")

    # @patch("requests.Session.post")
    # def test_node_remote_agent_failure(self, mock_post):
    #     """
    #     Test the `node_remote_agent` function for a failed response.

    #     This test mocks a failed HTTP POST request and verifies that
    #     the function handles exceptions appropriately.
    #     """
    #     # Mock a failed response from session.post
    #     mock_post.side_effect = Exception("Connection error")

    #     # Prepare test inputs
    #     messages = [HumanMessage(content="Hi")]
    #     github = {
    #         "github": {
    #             "repo_url": os.getenv("GITHUB_REPO_URL"),
    #             "github_token": os.getenv("GITHUB_TOKEN"),
    #             "branch": os.getenv("GITHUB_BRANCH"),
    #         }
    #     }
    #     thread_input = build_thread_input(messages=messages, extended_info=github)
    #     graph_config = RunnableConfig(
    #         configurable={
    #             "remote_agent": {
    #                 "id": "test_agent",
    #                 "url": "http://example.com",
    #                 "model": "gpt-4o",
    #                 "metadata": {},
    #             },
    #             "rest_timeout": 30,
    #         }
    #     )
    #     graph = build_graph(GraphConfig, GraphState)

    #     # Call the graph
    #     result = invoke_graph(thread_input=thread_input, config=graph_config, graph=graph)

    #     # Assertions
    #     self.assertEqual(result, "exception_node")
    #     self.assertIn("exception_msg", result.update)


# if __name__ == "__main__":
#    unittest.main()
