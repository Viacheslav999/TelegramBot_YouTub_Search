YouTube Downloader & AI Assistant Bot
🚀 About the Project
This is a Telegram bot that combines two powerful features:

YouTube Video Downloader: Easily download videos in various resolutions and formats directly from YouTube.
AI Assistant: Leverage the power of GPT-2 to answer questions and generate content.
The bot runs on a local server using Docker, making it easy to deploy and manage dependencies in an isolated environment.

Features
YouTube Video Downloading:

Search videos using a query or paste a direct YouTube link.
Choose from multiple resolutions (e.g., 360p, 720p, 1080p).
Downloads both video and audio streams and merges them seamlessly.
Converts videos to user-friendly aspect ratios and resolutions.
Optional upload to Mega cloud storage with public links.
AI Assistant:

Ask anything, and the bot will generate a response using GPT-2.
Logs conversations for analysis or improvement.
Interactive Telegram UI:

Inline keyboards for easy interaction.
Clean, intuitive design with helpful prompts.
Getting Started
Prerequisites
To run the bot locally, you'll need:

Docker installed (instructions here).
A Telegram Bot API Token (get it from BotFather).
A Mega Account for cloud uploads (optional).
FFmpeg installed inside the Docker container.
Installation
Clone the repository:

bash
Копировать код
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
Create a .env file: Add your credentials and paths:

env
Копировать код
TELEGRAM_BOT_TOKEN=your_bot_token
MEGA_EMAIL=your_mega_email
MEGA_PASSWORD=your_mega_password
FFMPEG_PATH=/usr/bin/ffmpeg  # Path inside the Docker container
Build and run the Docker container:

Build the Docker image:
bash
Копировать код
docker build -t youtube-ai-bot .
Run the container:
bash
Копировать код
docker run --name youtube-ai-bot -d --env-file .env youtube-ai-bot
Access the bot: Start the bot on Telegram by sending the /start command.

How to Use
Start the bot on Telegram with /start.
Choose an action:
Download a YouTube video.
Interact with the AI assistant.
Follow the inline prompts:
For YouTube: Send a link or search query, then select the desired resolution.
For AI: Enter a question or prompt and receive a response.
Receive your processed video or AI-generated reply!
Folder Structure
plaintext
Копировать код
project/
│
├── main.py                  # Main bot logic
├── Dockerfile               # Docker setup instructions
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not included in the repo)
├── videos/                  # Temporary folder for video storage
├── user_conversations.json  # Logs of AI interactions
└── README.md                # Project description
Technologies Used
aiogram: For building the Telegram bot.
yt-dlp: For YouTube downloading.
transformers: For AI text generation (GPT-2).
mega.py: For cloud storage integration.
FFmpeg: For video processing.
Docker: To package the bot and its dependencies into a containerized environment.
Dockerfile Example
The Dockerfile used to set up the project:

dockerfile
Копировать код
# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Set environment variables
ENV FFMPEG_PATH="/usr/bin/ffmpeg"

# Run the bot
CMD ["python", "main.py"]
Contributing
Contributions are welcome! Feel free to submit pull requests for bug fixes or new features.

License
This project is licensed under the MIT License.

Screenshots
You can add images or GIFs here to visually demonstrate the bot in action.

Example Commands
/start: Displays the main menu.
/link: Prompt to send a YouTube link or query.
AI Queries: Ask any question and receive a response.
