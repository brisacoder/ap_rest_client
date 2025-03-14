# AP Rest Client

AP Rest Client is a Python package that simplifies the process of making HTTP requests to RESTful APIs, specifically designed to interact with the Langchain Agent Protocol REST API. It provides a user-friendly interface for communicating with a remote agent running behind an AP Server, handling authentication, and parsing responses.

## Features

- Easy-to-use methods for GET, POST, PUT, DELETE requests
- Support for various authentication methods (e.g., API key, OAuth)
- Automatic parsing of JSON responses
- Customizable headers and parameters
- Error handling and logging

## Installation

You can install the package using pip:

```bash
pip install ap_rest_client
```

## Usage

Here is a basic example of how to use the AP Rest Client:

```python
from ap_rest_client.ap_protocol import invoke_graph

messages = [{"role": "user", "content": "Write a story about a cat"}]
print(invoke_graph(messages=messages))
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [your_email@example.com](mailto:your_email@example.com).

## References

- [Langchain Agent Protocol REST API](https://github.com/langchain-ai/agent-protocol)
