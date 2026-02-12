# YT Navigator

A project powered by AI to help you find information on specific YouTube channels. Allows users to search through video transcripts and interact with an AI assistant.

## Features

- **Video Processing**: Process individual YouTube videos or entire channels
- **Transcript Search**: Search through video content using advanced vector similarity
- **AI Chatbot**: Interact with videos using an intelligent chatbot
- **Channel Support**: Search within specific channels or across all videos
- **Modern UI**: Clean, responsive interface with dark theme

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.13+ (for local development)

### Using Docker Compose

```bash
docker compose build --no-cache server
docker compose up server
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

## Configuration

Environment variables are configured in `.env`:

- `GROQ_API_KEY`: Groq API key for LLM functionality
- `DATABASE_URL`: PostgreSQL connection string

## Usage

1. **Process Videos**: Add YouTube video or channel links to process content
2. **Search**: Use the query page to search through video transcripts
3. **Chat**: Interact with the AI chatbot for video insights

## Architecture

- **Backend**: Django with async support
- **Database**: PostgreSQL with vector extensions
- **AI**: Groq LLM with LangChain integration
- **Frontend**: Modern HTML with Tailwind CSS

## License

MIT License
