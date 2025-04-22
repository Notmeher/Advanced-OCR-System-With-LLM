from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import os
import io
import base64
from groq import Groq
from dotenv import load_dotenv
import tempfile
import fitz  # PyMuPDF
import secrets
import time

# Create Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client with better error handling
if not groq_api_key:
    print("WARNING: GROQ_API_KEY not found in environment variables!")
    client = None
else:
    try:
        client = Groq(api_key=groq_api_key)
        print("Groq client initialized successfully")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        client = None

def encode_image_to_base64(image):
    """Convert PIL Image to base64 encoding for API"""
    # Convert image to RGB mode to ensure JPEG compatibility
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
        
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def refine_text_with_llama(text, image, prompt_template):
    """Refine and contextualize extracted text using LLaMA Vision"""
    # Check if client is properly initialized
    if client is None:
        return "ERROR: Groq API client is not initialized. Please check your API key in the .env file."
    
    try:
        # Ensure image is in RGB mode for JPEG encoding
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
            
        img_base64 = encode_image_to_base64(image)
        
        # Prepare the prompt with the extracted text
        prompt = prompt_template.format(text=text)
        
        # Make the API call to LLaMA Vision with the updated model
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }],
            max_tokens=1024
        )
        
        return clean_output(chat_completion.choices[0].message.content)
    except Exception as e:
        return f"Error with LLaMA Vision API: {str(e)}\nPlease verify your Groq API key is correct and has permissions for this model."

def clean_output(text):
    """Remove explanatory phrases and clean up the output"""
    explanatory_phrases = [
        "I have extracted", 
        "The text in the image", 
        "As accurately as possible",
        "The image contains", 
        "I've transcribed", 
        "I've extracted",
        "The quality of the image",
        "The text appears to be",
        "From the image provided",
        "The content of the image",
        "Due to image quality",
        "Here is the text",
        "Text extraction complete",
        "Here's the extracted text",
        "The text from the image is",
        "I've maintained"
        "Here is the corrected text"

    ]
 
    # Split text into lines
    lines = text.split('\n')
    cleaned_lines = []
    
    # Process each line
    for line in lines:
        should_keep = True
        for phrase in explanatory_phrases:
            if phrase.lower() in line.lower():
                should_keep = False
                break
        
        if should_keep:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF file using PyMuPDF"""
    try:
        # Open the PDF file with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        images = []
        
        # Extract images from each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Create a high-resolution image of the page
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_bytes = pix.tobytes("jpeg")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        
        # Clean up
        pdf_document.close()
        
        return images, None
    except Exception as e:
        return None, f"Error extracting images from PDF: {str(e)}"

def process_file(file_path, file_type, page_selection, specific_page, prompt_template):
    """Process uploaded file (image or PDF)"""
    try:
        if not os.path.exists(file_path):
            return None, None, "File not found"
        
        if file_type == "image":
            # For image files
            image = Image.open(file_path)
            
            # Before calling the API, check if we have a valid client
            if client is None:
                return image, "ERROR: Groq API client is not initialized. Please check your API key in the .env file."
            
            refined_text = refine_text_with_llama("", image, prompt_template)
            
            # Save processed image to show on results page
            timestamp = int(time.time())
            image_preview_path = os.path.join(app.config['RESULTS_FOLDER'], f"preview_{timestamp}.jpg")
            image.save(image_preview_path)
            
            return image_preview_path, refined_text, None
        
        elif file_type == "pdf":
            # Process PDF
            images, error = extract_images_from_pdf(file_path)
            
            if error:
                return None, None, error
            
            # Check if client is initialized before processing
            if client is None:
                return None, None, "ERROR: Groq API client is not initialized. Please check your API key in the .env file."
            
            timestamp = int(time.time())
            
            if page_selection == "specific":
                # Process specific page
                if specific_page <= 0 or specific_page > len(images):
                    return None, None, f"Invalid page number. PDF has {len(images)} pages."
                
                # Process selected page
                image = images[specific_page - 1]
                refined_text = refine_text_with_llama("", image, prompt_template)
                
                # Save preview image
                image_preview_path = os.path.join(app.config['RESULTS_FOLDER'], f"preview_{timestamp}.jpg")
                image.save(image_preview_path)
                
                return image_preview_path, f"Page {specific_page}/{len(images)}\n\n{refined_text}", None
            
            else:  # Process all pages
                # Save first image as preview
                first_image = images[0]
                image_preview_path = os.path.join(app.config['RESULTS_FOLDER'], f"preview_{timestamp}.jpg")
                first_image.save(image_preview_path)
                
                # Process all pages
                all_text = f"PDF with {len(images)} pages\n\n"
                
                for i, img in enumerate(images):
                    print(f"Processing page {i+1}/{len(images)}")
                    refined = refine_text_with_llama("", img, prompt_template)
                    all_text += f"--- PAGE {i+1} ---\n{refined}\n\n"
                
                return image_preview_path, all_text, None
    
    except Exception as e:
        return None, None, f"Error processing file: {str(e)}"

def update_api_key(new_key):
    """Update the Groq API key"""
    global client
    if new_key.strip():
        try:
            client = Groq(api_key=new_key)
            return True, "API key updated successfully"
        except Exception as e:
            return False, f"Error updating API key: {str(e)}"
    return False, "No API key provided. Using environment variable if available."

# Flask routes
@app.route('/')
def index():
    api_status = "API Ready" if client is not None else "API Key Missing"
    return render_template('index.html', api_status=api_status)

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_type = request.form.get('file_type', 'image')
    page_selection = request.form.get('page_selection', 'specific')
    specific_page = int(request.form.get('specific_page', 1))
    
    # Use the provided prompt template instead of getting it from the form
    prompt_template = """I need to improve the text extracted using OCR from an image. Here's the raw OCR output:

Please analyze the image and correct any OCR errors. Pay special attention to:
1. Fixing misspelled words
2. Correcting misrecognized characters
3. Preserving the original formatting where possible
4. Handling any unclear or ambiguous text based on context
5. If you find any logo, don't process it
6. Only correct words if they are misspelled
7. Remove unnecessary explanation like PDF with 5 pages or Page Number 
8. Remove "*" 
Return only the corrected text without any explanations."""
    
    # Save uploaded file
    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Process the file
    image_path, result_text, error = process_file(
        file_path, file_type, page_selection, specific_page, prompt_template
    )
    
    if error:
        return jsonify({'error': error})
    
    # Save results to text file
    result_file_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{timestamp}.txt")
    with open(result_file_path, 'w', encoding='utf-8') as f:
        f.write(result_text)
    
    return jsonify({
        'image_path': image_path.replace('\\', '/') if image_path else None,
        'result_text': result_text,
        'result_file': result_file_path.replace('\\', '/')
    })

@app.route('/update_api_key', methods=['POST'])
def api_key_update():
    new_key = request.form.get('api_key', '')
    success, message = update_api_key(new_key)
    return jsonify({'success': success, 'message': message})

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/results/<path:filename>')
def result_file(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)