import streamlit as st
import os
import time
import torch
from PIL import Image
from pathlib import Path

# Import for Transformers approach
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    docling_available = True
except ImportError:
    docling_available = False


def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    if not transformers_available:
        missing.append("transformers")
    if not docling_available:
        missing.append("docling-core")
    
    return missing


def download_model_if_needed():
    """Download the model if it's not already downloaded"""
    try:
        # This will download the model if it's not already downloaded
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained("ds4sd/SmolDocling-256M-preview")
        return True
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False


def process_single_image(image, prompt_text="Convert code to text."):
    """Process a single image using locally downloaded model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")
    
    start_time = time.time()
    
    # Load processor and model
    try:
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.float32,
        ).to(device)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise
    
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    
    # Clean the output
    doctags = doctags.replace("<end_of_utterance>", "").strip()
    
    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    
    # Create a docling document
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    
    # Export as markdown
    md_content = doc.export_to_markdown()
    
    processing_time = time.time() - start_time
    
    return md_content, processing_time


def main():
    st.set_page_config(page_title="Local SmolDocling Code OCR App", layout="wide")
    
    st.title("SmolDocling Code to Text Converter")
    st.write("Upload code images to extract text using locally downloaded SmolDocling model")
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}. Please install them to use this app.")
        st.info("Install with: pip install " + " ".join(missing_deps))
        st.stop()
    
    # Download model if needed with progress indicator
    with st.spinner("Checking model availability (will download if needed)..."):
        model_ready = download_model_if_needed()
    
    if not model_ready:
        st.error("Failed to download or load the model. Please check your internet connection and try again.")
        st.stop()
    else:
        st.success("Model is ready to use!")
    
    # Create sidebar
    with st.sidebar:
        st.header("Input Options")
        
        upload_option = st.radio("Choose upload option:", ["Single Image", "Multiple Images"])
        
        if upload_option == "Single Image":
            uploaded_file = st.file_uploader("Upload code image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=250)
        else:
            uploaded_files = st.file_uploader("Upload multiple code images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Main content area
    if upload_option == "Single Image" and 'uploaded_file' in locals() and uploaded_file is not None:
        process_button = st.button("Process Image")
        
        if process_button:
            with st.spinner("Processing image..."):
                try:
                    md_content, processing_time = process_single_image(image, "Convert code to text.")
                    
                    st.subheader("Markdown Output")
                    st.markdown(md_content)
                    st.download_button("Download Markdown", md_content, file_name="code_text.md")
                    
                    st.success(f"Processing completed in {processing_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    elif upload_option == "Multiple Images" and 'uploaded_files' in locals() and uploaded_files:
        images = [Image.open(file).convert("RGB") for file in uploaded_files]
        
        if len(images) > 0:
            process_button = st.button("Process Images")
            
            if process_button:
                progress_bar = st.progress(0)
                with st.spinner(f"Processing {len(images)} images..."):
                    try:
                        # Process one by one
                        results = []
                        for idx, image in enumerate(images):
                            status_text = st.empty()
                            status_text.write(f"Processing image {idx+1}/{len(images)}...")
                            
                            md_content, processing_time = process_single_image(image, "Convert code to text.")
                            
                            results.append((md_content, processing_time))
                            progress_bar.progress((idx + 1) / len(images))
                        
                        for idx, (md_content, proc_time) in enumerate(results):
                            with st.expander(f"Image {idx+1} Results"):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.image(images[idx], caption=f"Image {idx+1}", width=250)
                                
                                with col2:
                                    st.markdown(md_content)
                                    st.download_button(f"Download Markdown {idx+1}", md_content, file_name=f"code_text_{idx+1}.md")
                                
                                st.write(f"Image {idx+1} processed in {proc_time:.2f} seconds")
                        
                        st.success(f"All images processed successfully")
                    except Exception as e:
                        st.error(f"Error processing images: {str(e)}")


if __name__ == "__main__":
    main()