// Google Material Design Chat Interface - Production Ready
// Comprehensive JavaScript for Dialogflow integration with advanced features

'use strict';

// Global state management
const AppState = {
    currentConversationId: null,
    conversations: JSON.parse(localStorage.getItem('conversations') || '[]'),
    attachedFile: null,
    screenshotData: null,
    cameraPhotoData: null,
    mediaRecorder: null,
    recordingChunks: [],
    isRecording: false,
    isPaused: false,
    cameraStream: null,
    screenStream: null,
    isTyping: false,
    messageHistory: [],
    userPreferences: JSON.parse(localStorage.getItem('userPreferences') || '{}')
};

// Utility functions
const Utils = {
    generateId: () => Math.random().toString(36).substr(2, 9),
    
    formatTime: (date) => {
        return new Intl.DateTimeFormat('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }).format(date);
    },

    formatFileSize: (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    sanitizeHTML: (str) => {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },

    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    throttle: (func, limit) => {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    },

    copyToClipboard: async (text) => {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            console.error('Failed to copy text: ', err);
            return false;
        }
    },

    downloadFile: (content, filename, contentType = 'text/plain') => {
        const blob = new Blob([content], { type: contentType });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    },

    announceToScreenReader: (message) => {
        const announcements = document.getElementById('announcements');
        if (announcements) {
            announcements.textContent = message;
            setTimeout(() => {
                announcements.textContent = '';
            }, 1000);
        }
    }
};

// Toast notification system
const Toast = {
    show: (message, type = 'info', duration = 5000) => {
        const container = document.getElementById('toast-container');
        const toastId = Utils.generateId();
        
        const typeIcons = {
            info: 'fas fa-info-circle',
            success: 'fas fa-check-circle',
            warning: 'fas fa-exclamation-triangle',
            error: 'fas fa-times-circle'
        };

        const typeColors = {
            info: 'border-google-blue',
            success: 'border-google-green', 
            warning: 'border-google-yellow',
            error: 'border-google-red'
        };

        const toast = document.createElement('div');
        toast.id = toastId;
        toast.className = `toast-notification bg-white rounded-lg shadow-lg p-4 mb-2 border-l-4 ${typeColors[type]} animate-slide-down`;
        toast.innerHTML = `
            <div class="flex items-start">
                <i class="${typeIcons[type]} text-lg mr-3 mt-0.5" style="color: var(--google-${type === 'info' ? 'blue' : type === 'success' ? 'green' : type === 'warning' ? 'yellow' : 'red'})"></i>
                <div class="flex-1">
                    <p class="text-sm text-google-dark-gray">${Utils.sanitizeHTML(message)}</p>
                </div>
                <button onclick="Toast.remove('${toastId}')" class="ml-3 text-google-gray hover:text-google-red focus:outline-none">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        container.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => Toast.remove(toastId), duration);
        }

        Utils.announceToScreenReader(message);
        return toastId;
    },

    remove: (toastId) => {
        const toast = document.getElementById(toastId);
        if (toast) {
            toast.style.animation = 'slideInUp 0.3s ease-out reverse';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }
    }
};

// Conversation management
const ConversationManager = {
    create: () => {
        const conversation = {
            id: Utils.generateId(),
            title: 'New Conversation',
            messages: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        
        AppState.conversations.unshift(conversation);
        AppState.currentConversationId = conversation.id;
        ConversationManager.save();
        ConversationManager.updateUI();
        ConversationManager.clearMessages();
        
        return conversation;
    },

    save: () => {
        localStorage.setItem('conversations', JSON.stringify(AppState.conversations));
    },

    load: (conversationId) => {
        const conversation = AppState.conversations.find(c => c.id === conversationId);
        if (!conversation) return null;

        AppState.currentConversationId = conversationId;
        ConversationManager.updateUI();
        ConversationManager.loadMessages(conversation.messages);
        
        return conversation;
    },

    addMessage: (message) => {
        const conversation = AppState.conversations.find(c => c.id === AppState.currentConversationId);
        if (conversation) {
            conversation.messages.push(message);
            conversation.updatedAt = new Date().toISOString();
            
            if (conversation.messages.length === 1 && message.sender === 'user') {
                conversation.title = message.content.substring(0, 50) + (message.content.length > 50 ? '...' : '');
            }
            
            ConversationManager.save();
            ConversationManager.updateUI();
        }
    },

    updateUI: () => {
        const historyContainer = document.getElementById('chat-history');
        const emptyState = document.getElementById('empty-history');
        const titleElement = document.getElementById('current-chat-title');

        historyContainer.innerHTML = '';

        if (AppState.conversations.length === 0) {
            emptyState.classList.remove('hidden');
            titleElement.textContent = 'New Conversation';
            return;
        }

        emptyState.classList.add('hidden');

        AppState.conversations.forEach(conversation => {
            const item = document.createElement('div');
            item.className = `chat-item ${conversation.id === AppState.currentConversationId ? 'active' : ''}`;
            item.innerHTML = `
                <div class="flex-1 min-w-0">
                    <div class="text-sm font-medium text-white truncate">
                        ${Utils.sanitizeHTML(conversation.title)}
                    </div>
                    <div class="text-xs text-white/60 mt-1">
                        ${Utils.formatTime(new Date(conversation.updatedAt))}
                    </div>
                </div>
            `;
            
            item.addEventListener('click', () => ConversationManager.load(conversation.id));
            historyContainer.appendChild(item);
        });

        const currentConversation = AppState.conversations.find(c => c.id === AppState.currentConversationId);
        if (currentConversation) {
            titleElement.textContent = currentConversation.title;
        }
    },

    loadMessages: (messages) => {
        ConversationManager.clearMessages();
        messages.forEach(message => {
            MessageManager.display(message.content, message.sender, message.timestamp, false);
        });
    },

    clearMessages: () => {
        const container = document.getElementById('messages-container');
        // Keep only the welcome message
        const welcomeMessage = container.querySelector('.bot-message');
        container.innerHTML = '';
        if (welcomeMessage) {
            container.appendChild(welcomeMessage);
        }
    },

    reset: () => {
        if (confirm('Are you sure you want to delete all conversations? This action cannot be undone.')) {
            AppState.conversations = [];
            AppState.currentConversationId = null;
            ConversationManager.save();
            ConversationManager.updateUI();
            ConversationManager.clearMessages();
            Toast.show('All conversations have been deleted', 'success');
        }
    },

    export: () => {
        const conversation = AppState.conversations.find(c => c.id === AppState.currentConversationId);
        if (!conversation) {
            Toast.show('No conversation to export', 'warning');
            return;
        }

        let transcript = `Conversation: ${conversation.title}\n`;
        transcript += `Created: ${new Date(conversation.createdAt).toLocaleString()}\n\n`;

        conversation.messages.forEach(message => {
            const timestamp = new Date(message.timestamp).toLocaleString();
            transcript += `[${timestamp}] ${message.sender.toUpperCase()}: ${message.content}\n\n`;
        });

        Utils.downloadFile(transcript, `conversation-${conversation.id}.txt`, 'text/plain');
        Toast.show('Conversation exported successfully', 'success');
    }
};

// Message management
const MessageManager = {
    display: (content, sender, timestamp = new Date(), animate = true) => {
        const container = document.getElementById('messages-container');
        const messageDiv = document.createElement('div');
        const timeStr = Utils.formatTime(new Date(timestamp));
        
        messageDiv.className = `flex items-start gap-4 ${sender}-message`;
        if (animate) {
            messageDiv.classList.add('animate-fade-in');
        }

        if (sender === 'user') {
            messageDiv.innerHTML = `
                <div class="w-8 h-8 bg-google-blue rounded-full flex items-center justify-center flex-shrink-0 user-avatar">
                    <i class="fas fa-user text-white text-sm"></i>
                </div>
                <div class="flex-1">
                    <div class="bg-white rounded-lg p-3 google-shadow">
                        <p class="text-google-dark-gray">${Utils.sanitizeHTML(content)}</p>
                    </div>
                    <div class="text-xs text-google-gray mt-2">
                        <time datetime="${timestamp}">${timeStr}</time>
                    </div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="avatar-gradient w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0">
                    <i class="fab fa-google text-white text-sm"></i>
                </div>
                <div class="flex-1">
                    <div class="prose max-w-none">
                        ${MessageManager.parseMarkdown(content)}
                    </div>
                    <div class="text-xs text-google-gray mt-2 flex items-center gap-3">
                        <time datetime="${timestamp}">${timeStr}</time>
                        <button onclick="MessageManager.copyMessage(this)" 
                                class="text-google-gray hover:text-google-blue transition-colors text-xs"
                                title="Copy message">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                        <button onclick="MessageManager.speakMessage(this)" 
                                class="text-google-gray hover:text-google-blue transition-colors text-xs"
                                title="Read aloud">
                            <i class="fas fa-volume-up"></i> Speak
                        </button>
                    </div>
                </div>
            `;
        }

        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;

        return messageDiv;
    },

    parseMarkdown: (content) => {
        // Simple markdown parsing for basic formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
            .replace(/\n/g, '<br>');
    },

    copyMessage: async (button) => {
        const messageDiv = button.closest('.bot-message, .user-message');
        const content = messageDiv.querySelector('.prose, p').textContent;
        
        const success = await Utils.copyToClipboard(content);
        if (success) {
            Toast.show('Message copied to clipboard', 'success');
            button.innerHTML = '<i class="fas fa-check"></i> Copied';
            setTimeout(() => {
                button.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        } else {
            Toast.show('Failed to copy message', 'error');
        }
    },

    speakMessage: (button) => {
        const messageDiv = button.closest('.bot-message, .user-message');
        const content = messageDiv.querySelector('.prose, p').textContent;
        
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(content);
            utterance.rate = 0.9;
            utterance.pitch = 1;
            utterance.volume = 0.8;
            
            utterance.onstart = () => {
                button.innerHTML = '<i class="fas fa-stop"></i> Stop';
                button.onclick = () => {
                    window.speechSynthesis.cancel();
                    button.innerHTML = '<i class="fas fa-volume-up"></i> Speak';
                    button.onclick = () => MessageManager.speakMessage(button);
                };
            };
            
            utterance.onend = () => {
                button.innerHTML = '<i class="fas fa-volume-up"></i> Speak';
                button.onclick = () => MessageManager.speakMessage(button);
            };
            
            window.speechSynthesis.speak(utterance);
        } else {
            Toast.show('Text-to-speech is not supported in your browser', 'warning');
        }
    }
};

// Input handling
const InputManager = {
    init: () => {
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const charCounter = document.getElementById('char-counter');
        const charCount = document.getElementById('char-count');

        messageInput.addEventListener('input', Utils.debounce(() => {
            InputManager.updateCharCount();
            InputManager.updateSendButton();
            InputManager.autoResize();
        }, 100));

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                InputManager.sendMessage();
            }
        });

        sendButton.addEventListener('click', InputManager.sendMessage);
    },

    updateCharCount: () => {
        const messageInput = document.getElementById('message-input');
        const charCount = document.getElementById('char-count');
        const charCounter = document.getElementById('char-counter');
        
        const currentLength = messageInput.value.length;
        const maxLength = 4000;
        
        charCount.textContent = currentLength;
        
        if (currentLength > maxLength * 0.9) {
            charCounter.className = 'absolute bottom-1 left-3 text-xs text-google-red';
        } else if (currentLength > maxLength * 0.7) {
            charCounter.className = 'absolute bottom-1 left-3 text-xs text-google-yellow';
        } else {
            charCounter.className = 'absolute bottom-1 left-3 text-xs text-google-gray';
        }
    },

    updateSendButton: () => {
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        
        const hasContent = messageInput.value.trim().length > 0;
        const hasAttachment = AppState.attachedFile || AppState.screenshotData || AppState.cameraPhotoData;
        
        sendButton.disabled = !hasContent && !hasAttachment;
    },

    autoResize: () => {
        const messageInput = document.getElementById('message-input');
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
    },

    sendMessage: async () => {
        const messageInput = document.getElementById('message-input');
        const message = messageInput.value.trim();
        
        if (!message && !AppState.attachedFile && !AppState.screenshotData && !AppState.cameraPhotoData) return;

        // Create conversation if none exists
        if (!AppState.currentConversationId) {
            ConversationManager.create();
        }

        // Display user message
        if (message) {
            const userMessage = {
                content: message,
                sender: 'user',
                timestamp: new Date().toISOString()
            };
            
            MessageManager.display(message, 'user');
            ConversationManager.addMessage(userMessage);
        }

        // Clear input
        messageInput.value = '';
        InputManager.updateCharCount();
        InputManager.updateSendButton();
        InputManager.autoResize();

        // Show typing indicator
        TypingIndicator.show();

        try {
            const response = await APIManager.sendMessage(message);
            TypingIndicator.hide();
            
            if (response.response) {
                const botMessage = {
                    content: response.response,
                    sender: 'bot',
                    timestamp: new Date().toISOString()
                };
                
                MessageManager.display(response.response, 'bot');
                ConversationManager.addMessage(botMessage);
            } else if (response.error) {
                Toast.show(response.error, 'error');
            }
        } catch (error) {
            TypingIndicator.hide();
            console.error('Error sending message:', error);
            Toast.show('Failed to send message. Please try again.', 'error');
        }

        // Clear attachments
        AttachmentManager.clear();
        ScreenshotManager.clear();
        CameraManager.clearPhoto();
    }
};

// API management
const APIManager = {
    sendMessage: async (message) => {
        const formData = {
            message: message || '',
            attachment: null
        };

        // Handle file attachment
        if (AppState.attachedFile) {
            formData.attachment = {
                filename: AppState.attachedFile.filename,
                isScreenshot: false
            };
        }

        // Handle screenshot
        if (AppState.screenshotData) {
            formData.attachment = {
                isScreenshot: true,
                data: AppState.screenshotData
            };
        }

        // Handle camera photo
        if (AppState.cameraPhotoData) {
            formData.attachment = {
                isScreenshot: true,
                data: AppState.cameraPhotoData
            };
        }

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    },

    uploadFile: async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    },

    uploadScreenshot: async (screenshotData) => {
        const response = await fetch('/upload-screenshot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ screenshot: screenshotData })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }
};

// Typing indicator
const TypingIndicator = {
    show: () => {
        document.getElementById('typing-container').classList.remove('hidden');
        const container = document.getElementById('messages-container');
        container.scrollTop = container.scrollHeight;
    },

    hide: () => {
        document.getElementById('typing-container').classList.add('hidden');
    }
};

// File attachment management
const AttachmentManager = {
    init: () => {
        const attachmentButton = document.getElementById('attachment-button');
        const fileInput = document.getElementById('file-input');
        const removeButton = document.getElementById('remove-attachment');

        attachmentButton.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', AttachmentManager.handleFileSelect);
        removeButton.addEventListener('click', AttachmentManager.clear);
    },

    handleFileSelect: async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            Toast.show('File size must be less than 16MB', 'error');
            return;
        }

        try {
            const response = await APIManager.uploadFile(file);
            
            if (response.success) {
                AppState.attachedFile = {
                    filename: response.filename,
                    originalName: response.original_filename,
                    size: file.size,
                    type: response.file_type,
                    url: response.url
                };

                AttachmentManager.showPreview();
                InputManager.updateSendButton();
                Toast.show('File attached successfully', 'success');
            } else {
                Toast.show(response.error || 'Failed to upload file', 'error');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            Toast.show('Failed to upload file', 'error');
        }

        // Clear file input
        event.target.value = '';
    },

    showPreview: () => {
        if (!AppState.attachedFile) return;

        const preview = document.getElementById('attachment-preview');
        const nameElement = document.getElementById('attachment-name');
        const sizeElement = document.getElementById('attachment-size');

        nameElement.textContent = AppState.attachedFile.originalName;
        sizeElement.textContent = Utils.formatFileSize(AppState.attachedFile.size);
        
        preview.classList.remove('hidden');
    },

    clear: () => {
        AppState.attachedFile = null;
        document.getElementById('attachment-preview').classList.add('hidden');
        InputManager.updateSendButton();
    }
};

// Screenshot management
const ScreenshotManager = {
    init: () => {
        const screenshotButton = document.getElementById('screenshot-button');
        const removeButton = document.getElementById('remove-screenshot');

        screenshotButton.addEventListener('click', ScreenshotManager.openModal);
        removeButton.addEventListener('click', ScreenshotManager.clear);

        // Modal buttons
        document.getElementById('capture-visible-area').addEventListener('click', () => {
            ScreenshotManager.captureVisibleArea();
            ModalManager.close('full-screenshot-modal');
        });

        document.getElementById('capture-full-page').addEventListener('click', () => {
            ScreenshotManager.captureFullPage();
            ModalManager.close('full-screenshot-modal');
        });
    },

    openModal: () => {
        ModalManager.open('full-screenshot-modal');
    },

    captureVisibleArea: async () => {
        try {
            const canvas = await html2canvas(document.body, {
                height: window.innerHeight,
                width: window.innerWidth,
                scrollX: 0,
                scrollY: 0
            });
            
            const dataURL = canvas.toDataURL('image/png');
            ScreenshotManager.handleScreenshot(dataURL);
        } catch (error) {
            console.error('Error capturing visible area:', error);
            Toast.show('Failed to capture screenshot', 'error');
        }
    },

    captureFullPage: async () => {
        try {
            const canvas = await html2canvas(document.body);
            const dataURL = canvas.toDataURL('image/png');
            ScreenshotManager.handleScreenshot(dataURL);
        } catch (error) {
            console.error('Error capturing full page:', error);
            Toast.show('Failed to capture screenshot', 'error');
        }
    },

    handleScreenshot: (dataURL) => {
        AppState.screenshotData = dataURL;
        ScreenshotManager.showPreview(dataURL);
        InputManager.updateSendButton();
        Toast.show('Screenshot captured successfully', 'success');
    },

    showPreview: (dataURL) => {
        const preview = document.getElementById('screenshot-preview');
        const thumbnail = document.getElementById('screenshot-thumbnail');
        const sizeElement = document.getElementById('screenshot-size');

        thumbnail.src = dataURL;
        
        // Estimate file size (base64 data is roughly 4/3 the size of the original)
        const estimatedSize = Math.round((dataURL.length * 3) / 4);
        sizeElement.textContent = Utils.formatFileSize(estimatedSize);
        
        preview.classList.remove('hidden');
    },

    clear: () => {
        AppState.screenshotData = null;
        document.getElementById('screenshot-preview').classList.add('hidden');
        InputManager.updateSendButton();
    }
};

// Camera management
const CameraManager = {
    init: () => {
        const cameraButton = document.getElementById('camera-button');
        const removeCameraPhoto = document.getElementById('remove-camera-photo');

        cameraButton.addEventListener('click', CameraManager.openModal);
        removeCameraPhoto.addEventListener('click', CameraManager.clearPhoto);

        // Modal buttons
        document.getElementById('start-camera').addEventListener('click', CameraManager.startCamera);
        document.getElementById('take-photo').addEventListener('click', CameraManager.takePhoto);
        document.getElementById('retake-photo').addEventListener('click', CameraManager.retakePhoto);
        document.getElementById('stop-camera').addEventListener('click', CameraManager.stopCamera);
        document.getElementById('send-camera-photo').addEventListener('click', CameraManager.sendPhoto);
    },

    openModal: () => {
        ModalManager.open('camera-modal');
        CameraManager.resetModal();
    },

    resetModal: () => {
        // Hide all elements except start button
        document.getElementById('camera-placeholder').classList.remove('hidden');
        document.getElementById('camera-video').classList.add('hidden');
        document.getElementById('camera-canvas').classList.add('hidden');
        document.getElementById('captured-photo-container').classList.add('hidden');
        
        document.getElementById('start-camera').classList.remove('hidden');
        document.getElementById('take-photo').classList.add('hidden');
        document.getElementById('retake-photo').classList.add('hidden');
        document.getElementById('stop-camera').classList.add('hidden');
        document.getElementById('send-camera-photo').classList.add('hidden');
        
        document.getElementById('camera-status').textContent = 'Ready to start camera';
    },

    startCamera: async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            
            AppState.cameraStream = stream;
            const video = document.getElementById('camera-video');
            video.srcObject = stream;
            
            // Show video and controls
            document.getElementById('camera-placeholder').classList.add('hidden');
            document.getElementById('camera-video').classList.remove('hidden');
            
            document.getElementById('start-camera').classList.add('hidden');
            document.getElementById('take-photo').classList.remove('hidden');
            document.getElementById('stop-camera').classList.remove('hidden');
            
            document.getElementById('camera-status').textContent = 'Camera active - ready to take photo';
            Toast.show('Camera started successfully', 'success');
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            Toast.show('Failed to access camera. Please ensure you have granted camera permissions.', 'error');
            document.getElementById('camera-status').textContent = 'Failed to start camera';
        }
    },

    takePhoto: () => {
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');
        const context = canvas.getContext('2d');
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get image data
        const dataURL = canvas.toDataURL('image/png');
        
        // Show captured photo
        const capturedPhoto = document.getElementById('captured-photo');
        capturedPhoto.src = dataURL;
        
        // Update UI
        document.getElementById('camera-video').classList.add('hidden');
        document.getElementById('captured-photo-container').classList.remove('hidden');
        
        document.getElementById('take-photo').classList.add('hidden');
        document.getElementById('retake-photo').classList.remove('hidden');
        document.getElementById('send-camera-photo').classList.remove('hidden');
        document.getElementById('send-camera-photo').disabled = false;
        
        document.getElementById('camera-status').textContent = 'Photo captured - review and send';
        AppState.cameraPhotoData = dataURL;
        
        Toast.show('Photo captured successfully', 'success');
    },

    retakePhoto: () => {
        // Show video again
        document.getElementById('camera-video').classList.remove('hidden');
        document.getElementById('captured-photo-container').classList.add('hidden');
        
        document.getElementById('take-photo').classList.remove('hidden');
        document.getElementById('retake-photo').classList.add('hidden');
        document.getElementById('send-camera-photo').classList.add('hidden');
        
        document.getElementById('camera-status').textContent = 'Camera active - ready to take photo';
        AppState.cameraPhotoData = null;
    },

    stopCamera: () => {
        if (AppState.cameraStream) {
            AppState.cameraStream.getTracks().forEach(track => track.stop());
            AppState.cameraStream = null;
        }
        
        CameraManager.resetModal();
        Toast.show('Camera stopped', 'info');
    },

    sendPhoto: () => {
        if (AppState.cameraPhotoData) {
            ModalManager.close('camera-modal');
            CameraManager.showPhotoPreview();
            InputManager.updateSendButton();
            CameraManager.stopCamera();
            Toast.show('Photo ready to send', 'success');
        }
    },

    showPhotoPreview: () => {
        const preview = document.getElementById('camera-preview');
        const thumbnail = document.getElementById('camera-photo-thumbnail');
        const sizeElement = document.getElementById('camera-photo-size');

        thumbnail.src = AppState.cameraPhotoData;
        
        // Estimate file size
        const estimatedSize = Math.round((AppState.cameraPhotoData.length * 3) / 4);
        sizeElement.textContent = Utils.formatFileSize(estimatedSize);
        
        preview.classList.remove('hidden');
    },

    clearPhoto: () => {
        AppState.cameraPhotoData = null;
        document.getElementById('camera-preview').classList.add('hidden');
        InputManager.updateSendButton();
    }
};

// Screen capture management  
const ScreenCaptureManager = {
    init: () => {
        const screenCaptureButton = document.getElementById('screen-capture-button');
        
        screenCaptureButton.addEventListener('click', ScreenCaptureManager.openModal);

        // Modal buttons
        document.getElementById('start-screen-capture').addEventListener('click', ScreenCaptureManager.startCapture);
        document.getElementById('capture-screenshot').addEventListener('click', ScreenCaptureManager.captureScreenshot);
        document.getElementById('stop-screen-capture').addEventListener('click', ScreenCaptureManager.stopCapture);
        document.getElementById('send-screen-capture').addEventListener('click', ScreenCaptureManager.sendCapture);
    },

    openModal: () => {
        ModalManager.open('screen-capture-modal');
        ScreenCaptureManager.resetModal();
    },

    resetModal: () => {
        document.getElementById('screen-capture-placeholder').classList.remove('hidden');
        document.getElementById('screen-preview').classList.add('hidden');
        document.getElementById('screen-canvas').classList.add('hidden');
        
        document.getElementById('start-screen-capture').classList.remove('hidden');
        document.getElementById('capture-screenshot').classList.add('hidden');
        document.getElementById('stop-screen-capture').classList.add('hidden');
        document.getElementById('send-screen-capture').classList.add('hidden');
        
        document.getElementById('screen-capture-status').textContent = 'Ready to capture screen';
    },

    startCapture: async () => {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: { mediaSource: 'screen' }
            });
            
            AppState.screenStream = stream;
            const video = document.getElementById('screen-preview');
            video.srcObject = stream;
            
            document.getElementById('screen-capture-placeholder').classList.add('hidden');
            document.getElementById('screen-preview').classList.remove('hidden');
            
            document.getElementById('start-screen-capture').classList.add('hidden');
            document.getElementById('capture-screenshot').classList.remove('hidden');
            document.getElementById('stop-screen-capture').classList.remove('hidden');
            
            document.getElementById('screen-capture-status').textContent = 'Screen sharing active - ready to capture';
            Toast.show('Screen sharing started', 'success');
            
        } catch (error) {
            console.error('Error starting screen capture:', error);
            Toast.show('Failed to start screen sharing', 'error');
        }
    },

    captureScreenshot: () => {
        const video = document.getElementById('screen-preview');
        const canvas = document.getElementById('screen-canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/png');
        
        document.getElementById('screen-preview').classList.add('hidden');
        document.getElementById('screen-canvas').classList.remove('hidden');
        
        document.getElementById('capture-screenshot').classList.add('hidden');
        document.getElementById('send-screen-capture').classList.remove('hidden');
        document.getElementById('send-screen-capture').disabled = false;
        
        document.getElementById('screen-capture-status').textContent = 'Screenshot captured - ready to send';
        AppState.screenshotData = dataURL;
        
        Toast.show('Screenshot captured', 'success');
    },

    stopCapture: () => {
        if (AppState.screenStream) {
            AppState.screenStream.getTracks().forEach(track => track.stop());
            AppState.screenStream = null;
        }
        
        ScreenCaptureManager.resetModal();
        Toast.show('Screen sharing stopped', 'info');
    },

    sendCapture: () => {
        if (AppState.screenshotData) {
            ModalManager.close('screen-capture-modal');
            ScreenshotManager.showPreview(AppState.screenshotData);
            InputManager.updateSendButton();
            ScreenCaptureManager.stopCapture();
        }
    }
};

// Voice input management
const VoiceManager = {
    init: () => {
        const voiceButton = document.getElementById('voice-button');
        
        voiceButton.addEventListener('click', VoiceManager.openModal);

        // Modal buttons
        document.getElementById('start-recording').addEventListener('click', VoiceManager.toggleRecording);
        document.getElementById('pause-recording').addEventListener('click', VoiceManager.pauseRecording);
        document.getElementById('resume-recording').addEventListener('click', VoiceManager.resumeRecording);
        document.getElementById('reset-recording').addEventListener('click', VoiceManager.resetRecording);
        document.getElementById('send-voice').addEventListener('click', VoiceManager.sendVoice);
    },

    openModal: () => {
        ModalManager.open('voice-modal');
        VoiceManager.resetModal();
    },

    resetModal: () => {
        document.getElementById('transcription-result').textContent = 'Transcription will appear here...';
        document.getElementById('recording-status').textContent = 'Click the microphone to start recording';
        
        document.getElementById('pause-recording').disabled = true;
        document.getElementById('resume-recording').disabled = true;
        document.getElementById('reset-recording').disabled = true;
        document.getElementById('send-voice').disabled = true;
        
        AppState.isRecording = false;
        AppState.isPaused = false;
    },

    toggleRecording: async () => {
        if (!AppState.isRecording) {
            await VoiceManager.startRecording();
        } else {
            VoiceManager.stopRecording();
        }
    },

    startRecording: async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            AppState.mediaRecorder = new MediaRecorder(stream);
            AppState.recordingChunks = [];
            AppState.isRecording = true;
            
            AppState.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    AppState.recordingChunks.push(event.data);
                }
            };
            
            AppState.mediaRecorder.onstop = VoiceManager.processRecording;
            AppState.mediaRecorder.start();
            
            VoiceManager.updateRecordingUI(true);
            Toast.show('Recording started', 'success');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            Toast.show('Failed to access microphone', 'error');
        }
    },

    stopRecording: () => {
        if (AppState.mediaRecorder && AppState.isRecording) {
            AppState.mediaRecorder.stop();
            AppState.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            AppState.isRecording = false;
            VoiceManager.updateRecordingUI(false);
        }
    },

    pauseRecording: () => {
        if (AppState.mediaRecorder && AppState.isRecording) {
            AppState.mediaRecorder.pause();
            AppState.isPaused = true;
            VoiceManager.updateRecordingUI(false);
        }
    },

    resumeRecording: () => {
        if (AppState.mediaRecorder && AppState.isPaused) {
            AppState.mediaRecorder.resume();
            AppState.isPaused = false;
            VoiceManager.updateRecordingUI(true);
        }
    },

    updateRecordingUI: (isRecording) => {
        const recordButton = document.getElementById('start-recording');
        const indicator = document.getElementById('recording-indicator');
        const status = document.getElementById('recording-status');
        
        if (isRecording) {
            recordButton.innerHTML = '<i class="fas fa-stop"></i>';
            recordButton.title = 'Stop recording';
            indicator.classList.remove('hidden');
            status.textContent = 'Recording... Click stop when finished';
            
            document.getElementById('pause-recording').disabled = false;
            document.getElementById('reset-recording').disabled = false;
        } else {
            recordButton.innerHTML = '<i class="fas fa-microphone"></i>';
            recordButton.title = 'Start recording';
            indicator.classList.add('hidden');
            status.textContent = AppState.isPaused ? 'Recording paused' : 'Click the microphone to start recording';
            
            document.getElementById('pause-recording').disabled = true;
            document.getElementById('resume-recording').disabled = !AppState.isPaused;
        }
    },

    processRecording: async () => {
        const audioBlob = new Blob(AppState.recordingChunks, { type: 'audio/wav' });
        
        // Simple transcription placeholder - in a real app you'd send to a speech-to-text service
        const transcriptionResult = document.getElementById('transcription-result');
        transcriptionResult.textContent = 'Voice recording ready to send (transcription not implemented in demo)';
        
        document.getElementById('send-voice').disabled = false;
        AppState.recordingChunks = [];
        
        Toast.show('Recording processed', 'success');
    },

    resetRecording: () => {
        VoiceManager.stopRecording();
        VoiceManager.resetModal();
    },

    sendVoice: () => {
        const transcription = document.getElementById('transcription-result').textContent;
        if (transcription && transcription !== 'Transcription will appear here...') {
            document.getElementById('message-input').value = transcription;
            ModalManager.close('voice-modal');
            InputManager.updateSendButton();
            Toast.show('Voice input added to message', 'success');
        }
    }
};

// Modal management
const ModalManager = {
    init: () => {
        // Close modal buttons
        document.querySelectorAll('[id^="close-"]').forEach(button => {
            button.addEventListener('click', (e) => {
                const modalId = e.target.closest('[role="dialog"]').id;
                ModalManager.close(modalId);
            });
        });

        // Close modals on backdrop click
        document.querySelectorAll('[role="dialog"]').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    ModalManager.close(modal.id);
                }
            });
        });

        // Close modals on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const openModal = document.querySelector('[role="dialog"]:not(.hidden)');
                if (openModal) {
                    ModalManager.close(openModal.id);
                }
            }
        });
    },

    open: (modalId) => {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
            
            // Focus the first focusable element
            const focusable = modal.querySelector('button, input, textarea, select');
            if (focusable) {
                focusable.focus();
            }
        }
    },

    close: (modalId) => {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = '';
            
            // Clean up any ongoing operations
            if (modalId === 'voice-modal') {
                VoiceManager.stopRecording();
            } else if (modalId === 'camera-modal') {
                CameraManager.stopCamera();
            } else if (modalId === 'screen-capture-modal') {
                ScreenCaptureManager.stopCapture();
            }
        }
    }
};

// UI interactions
const UIManager = {
    init: () => {
        // Menu toggle for mobile
        const menuToggle = document.getElementById('menu-toggle');
        const sidebar = document.getElementById('sidebar');
        
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            const overlay = UIManager.createOverlay();
            overlay.classList.add('active');
        });

        // New conversation
        document.getElementById('new-chat-btn').addEventListener('click', () => {
            ConversationManager.create();
        });

        // Reset conversations
        document.getElementById('reset-conversations-btn').addEventListener('click', () => {
            ConversationManager.reset();
        });

        // Download transcript
        document.getElementById('download-transcript-btn').addEventListener('click', () => {
            ConversationManager.export();
        });

        // More options menu
        const moreOptionsBtn = document.getElementById('more-options-btn');
        const moreOptionsMenu = document.getElementById('more-options-menu');
        
        moreOptionsBtn.addEventListener('click', () => {
            moreOptionsMenu.classList.toggle('hidden');
            moreOptionsBtn.setAttribute('aria-expanded', !moreOptionsMenu.classList.contains('hidden'));
        });

        // Close more options menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!moreOptionsBtn.contains(e.target) && !moreOptionsMenu.contains(e.target)) {
                moreOptionsMenu.classList.add('hidden');
                moreOptionsBtn.setAttribute('aria-expanded', 'false');
            }
        });

        // Privacy info
        document.getElementById('privacy-info').addEventListener('click', () => {
            Toast.show('This demo uses Google Dialogflow for conversation processing. No personal data is stored permanently.', 'info', 8000);
        });

        // Set welcome time
        document.getElementById('welcome-time').textContent = Utils.formatTime(new Date());
        document.getElementById('welcome-time').setAttribute('datetime', new Date().toISOString());
    },

    createOverlay: () => {
        let overlay = document.querySelector('.overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'overlay';
            overlay.addEventListener('click', () => {
                document.getElementById('sidebar').classList.remove('open');
                overlay.classList.remove('active');
            });
            document.body.appendChild(overlay);
        }
        return overlay;
    }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Hide loading screen
    setTimeout(() => {
        document.getElementById('loading-screen').style.opacity = '0';
        setTimeout(() => {
            document.getElementById('loading-screen').style.display = 'none';
        }, 300);
    }, 1000);

    // Initialize all components
    InputManager.init();
    AttachmentManager.init();
    ScreenshotManager.init();
    CameraManager.init();
    ScreenCaptureManager.init();
    VoiceManager.init();
    ModalManager.init();
    UIManager.init();
    
    // Load existing conversations
    ConversationManager.updateUI();
    
    // Create initial conversation if none exist
    if (AppState.conversations.length === 0) {
        ConversationManager.create();
    } else {
        // Load the most recent conversation
        ConversationManager.load(AppState.conversations[0].id);
    }

    Toast.show('Welcome to Google AI Assistant!', 'success');
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    // Stop any ongoing recordings or streams
    if (AppState.cameraStream) {
        AppState.cameraStream.getTracks().forEach(track => track.stop());
    }
    if (AppState.screenStream) {
        AppState.screenStream.getTracks().forEach(track => track.stop());
    }
    if (AppState.mediaRecorder && AppState.isRecording) {
        AppState.mediaRecorder.stop();
    }
});

// Error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    Toast.show('An unexpected error occurred. Please refresh the page if problems persist.', 'error');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    Toast.show('A network or processing error occurred. Please try again.', 'error');
});

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', () => {
        const loadTime = performance.now();
        console.log(`Application loaded in ${loadTime.toFixed(2)}ms`);
    });
}

// Export for global access (if needed)
window.AppState = AppState;
window.Toast = Toast;
window.Utils = Utils;