# 🧞‍♂️ Skill-Genie: AI-Powered Conversational Assistant

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)
![Dialogflow](https://img.shields.io/badge/Dialogflow-CX-orange.svg)

## 📋 Overview

Skill-Genie is an intelligent conversational assistant built with Google's Dialogflow CX and Flask. The application provides a modern, responsive chat interface where users can interact with an AI assistant through text or voice input, facilitating natural and engaging conversations.

## ✨ Features

- 🤖 **AI-Powered Conversations**: Leverages Google's Dialogflow CX for natural language understanding
- 🎙️ **Voice Input Support**: Speak directly to the assistant with speech-to-text capabilities
- 💬 **Chat History**: Maintains conversation history for easy reference
- 📎 **File Upload**: Support for attachments in conversations
- 📱 **Responsive Design**: Mobile-friendly interface using Tailwind CSS
- 🔊 **Text-to-Speech**: Voice responses for accessibility
- 🎨 **Modern UI**: Clean, intuitive interface with Accenture-inspired theming

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask 3.0.3 |
| NLU Engine | Google Dialogflow CX |
| Frontend | HTML, JavaScript, Tailwind CSS |
| Server | Gunicorn 23.0.0 |
| HTTP Handling | Werkzeug 3.0.3 |

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Google Cloud account with Dialogflow CX enabled
- Google Cloud credentials configured

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Yash-Kavaiya/Skill-genie.git
cd Skill-genie
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
python -m pip install -r requirements.txt
```

4. **Set up environment variables**

For Google Cloud authentication, you'll need to set up credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-credentials.json"
```

On Windows:
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=path\to\your-credentials.json
```

5. **Run the application**

```bash
python main.py
```

The application will be available at `http://localhost:8080`.

## 🧩 Project Structure

```
skill-genie/
├── main.py                # Flask application entry point
├── dialogflow_api.py      # Dialogflow CX integration
├── requirements.txt       # Python dependencies
├── static/                # Static assets
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── uploads/           # User uploads directory
└── templates/             # HTML templates
    ├── index.html         # Main chat interface
    ├── my-trips.html      # Trips page
    ├── destinations.html  # Destinations page
    └── ...                # Other page templates
```

## 🔧 Configuration

The application uses the following configurations:

- **Dialogflow Project**: "gen-ai-guru-gdg-pune"
- **Agent ID**: "ffff32e4-24a6-44de-9450-2475f80cc583"
- **File Upload Settings**:
  - Max File Size: 16MB
  - Supported Extensions: txt, pdf, png, jpg, jpeg, gif, doc, docx, xls, xlsx

To modify these configurations, update the settings in `main.py` and `dialogflow_api.py`.

## 🌐 Deployment

The application can be deployed to any cloud platform that supports Python applications. For Google Cloud Platform:

1. Make sure you have the gcloud CLI installed
2. Initialize your project:
```bash
gcloud init
```
3. Deploy to App Engine:
```bash
gcloud app deploy
```

## 💡 Usage Example

Once the application is running, you can:

1. Start a new conversation by clicking "New Conversation"
2. Type messages in the input field or use voice input
3. Attach files by clicking the paperclip icon
4. View and navigate through your conversation history

## 📚 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Google Cloud Dialogflow CX](https://cloud.google.com/dialogflow) for the conversational AI capabilities
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com/) for the UI styling
- [Font Awesome](https://fontawesome.com/) for the icons