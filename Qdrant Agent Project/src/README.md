## Overview

This project integrates various AI and machine learning tools to process and analyze oral argument transcripts from U.S. Federal Court Cases. The primary components include language models, vector stores, and a query processing function that utilizes these resources to provide detailed responses to legal queries.

## Prerequisites

- Python 3.8 or higher
- `dotenv` library for environment variable management
- Required libraries: 
    os
    python-dotenv
    logging
    openai
    qdrant-client
    langchain-openai
    llama-index
    chainlit
    langchain-core
    langchain-schema
    langsmith

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/cbrown1515/Court-Oral-Argument-Analyzer.git
    cd <CourtOralArgumentAnalyzer>
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    - Create a `.env` file in the project root directory.
    - Add the following variables with your respective API keys:
      ```
      OPENAI_API_KEY=your_openai_api_key
      QDRANT_API_KEY=your_qdrant_api_key
      NVIDIA_API_KEY=your_nvidia_api_key
      LANGCHAIN_API_KEY=your_langchain_api_key
      ```

## Project Structure

- `langbot3.py`: Main script to run the AI agent and handle chat interactions.
- `.env`: Environment variables file for API keys.
- `requirements.txt`: List of required Python packages.

## Contributing

Please feel free to contribute by submitting a pull request. Ensure your code adheres to the project guidelines and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to reach out with any questions or issues. Enjoy using the AI Legal Thinker!