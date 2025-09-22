import os
import base64
from google import genai
from google.genai import types
from PIL import Image
import io
from typing import Optional, Union, List, Dict
import speech_recognition as sr
import numpy as np
import soundfile as sf

class ITSupportAgent:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables. "
                "Get your API key from: https://aistudio.google.com/"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-exp"
        self.system_prompt = r"""You are an expert IT Support Agent with comprehensive knowledge in:
- Computer hardware and software troubleshooting
- Network and connectivity issues
- Operating systems (Windows, macOS, Linux)
- Software installation and configuration
- Security and malware issues
- Cloud services and applications
- Mobile devices support
- Printer and peripheral issues
- Data backup and recovery

## Response Format Guidelines:

**Always structure your responses using this format:**

1. **Start with a clear explanation** of what the error/issue means
2. **Use proper formatting** with bold headers, numbered lists, and bullet points
3. **Provide step-by-step solutions** with clear instructions
4. **Use visual indicators** like ✅ for solutions, ⚠️ for warnings, 📁 for file paths
5. **Include technical details** when relevant (file paths, registry keys, etc.)
6. **Offer multiple solutions** when applicable, ordered by likelihood of success

## Formatting Standards:

**For error messages:**
- **Quote the exact error** in the response
- **Explain what it means** in simple terms
- **Bold key terms** like software names, error codes, file types

**For solutions:**
- Use ✅ to mark the main solution section
- Number the steps (1., 2., 3., etc.)
- Use **bold text** for important actions like "Restart", "Uninstall", "Navigate to"
- Use `code formatting` for file paths, commands, and technical terms
- Use bullet points (*) for sub-steps or options

**For file paths:**
- Always use backticks: `C:\Program Files\Adobe\...`
- Include full paths when relevant
- Mention both Windows and Mac paths when applicable

**For warnings:**
- Use ⚠️ for important warnings
- **Bold** critical safety information
- Mention potential risks of each solution

## Example Response Structure:

**Q: [Restate the user's question clearly]**

**A:** [Brief explanation of what the error/issue means]

✅ **How to fix it:**

1. **First solution step**
   * Sub-step if needed
   * Additional details with `technical terms`

2. **Second solution step**
   * Navigate to: `C:\specific\file\path`
   * Specific instructions

3. **Alternative solution**
   * When to use this option
   * Step-by-step process

⚠️ **Important:** [Any warnings or considerations]

**Additional tips:**
* Prevention measures
* When to contact professional support

Your role is to:

1. Always answer in just 5 lines. Analyze the user's IT problem carefully by looking at error messages, screenshots, or descriptions
2. Provide clear, well-formatted step-by-step solutions
3. Explain technical concepts in simple terms while maintaining accuracy
4. Offer multiple solutions ordered by effectiveness and safety
5. Include relevant file paths, commands, or technical details
6. Warn about potential risks and suggest preventive measures
7. Be patient, professional, and thorough in your responses

Always format your responses to be visually clear and easy to follow, using the formatting guidelines above."""

    def process_image(self, image_data: bytes, mime_type: str) -> types.Part:
        """Process image data for Gemini API"""
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return types.Part.from_bytes(
            mime_type=mime_type,
            data=base64.b64decode(base64_image)
        )

    def process_text_file(self, file_content: str) -> str:
        """Process text file content"""
        return f"File Content:\n{file_content}\n"

    def process_audio(self, audio_data: bytes) -> str:
        """Process audio data to extract text using speech recognition"""
        try:
            # Convert audio bytes to numpy array
            audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # Initialize speech recognizer
            recognizer = sr.Recognizer()
            
            # Convert numpy array to audio data
            with io.BytesIO() as wav_io:
                sf.write(wav_io, audio_np, sample_rate, format='wav')
                wav_io.seek(0)
                with sr.AudioFile(wav_io) as source:
                    audio = recognizer.record(source)
            
            # Perform speech recognition
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            print(f"Speech recognition error: {str(e)}")
            return ""

    async def get_support_response(
        self, 
        text_query: str = "", 
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        file_content: Optional[str] = None
    ) -> str:
        """Get IT support response from Gemini with image and voice input"""
        
        contents = []
        user_parts = []
        
        # Process audio if provided
        if audio_data:
            voice_text = self.process_audio(audio_data)
            if voice_text:
                text_query = f"{text_query} {voice_text}".strip()
        
        # Add image if provided
        if image_data and image_mime_type:
            user_parts.append(self.process_image(image_data, image_mime_type))
        
        # Add file content if provided
        if file_content:
            query_with_file = f"{self.process_text_file(file_content)}\n\nUser Query: {text_query}"
            user_parts.append(types.Part(text=query_with_file))
        else:
            user_parts.append(types.Part(text=text_query))
        
        # Create content
        contents.append(types.Content(
            role="user",
            parts=user_parts
        ))
        
        # Generate content configuration
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part(text=self.system_prompt)
            ],
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2048
        )
        
        try:
            # Get response from Gemini
            response = await self.client.models.generate_content_async(
                model=self.model,
                contents=contents,
                config=generate_content_config
            )
            
            # Extract and return the response text
            if response and response.text:
                return response.text
            else:
                return "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question."

    def get_support_response_sync(
        self, 
        text_query: str = "", 
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> str:
        """Synchronous version of get_support_response"""
        
        contents = []
        
        # Build the user query
        user_parts = []
        
        # Add image if provided
        if image_data and image_mime_type:
            user_parts.append(self.process_image(image_data, image_mime_type))
        
        # Add file content if provided
        if file_content:
            query_with_file = f"{self.process_text_file(file_content)}\n\nUser Query: {text_query}"
            user_parts.append(types.Part(text=query_with_file))
        else:
            user_parts.append(types.Part(text=text_query if text_query else "Please help me with this IT issue shown in the image."))
        
        # Create content
        contents.append(types.Content(
            role="user",
            parts=user_parts
        ))
        
        # Generate content configuration
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part(text=self.system_prompt)
            ],
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2048
        )
        
        try:
            # Get response from Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config
            )
            
            # Extract and return the response text
            if response and response.text:
                return response.text
            else:
                return "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            return f"I encountered an error while processing your request. Please try again or contact support if the issue persists."