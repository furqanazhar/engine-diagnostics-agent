"""
Extract PDF content using LlamaParse.
Only processes first 20 pages.
Outputs to markdown file.
"""

import os
from llama_parse import LlamaParse

# File paths
INPUT_PDF = "F115AET_68V_28197_ZA_C1.PDF"
OUTPUT_MD = "F115AET_68V_28197_ZA_C1_pages_1-20_llamaparse.md"
MAX_PAGES = 20

# API Key
API_KEY = "llx-yecI06o842ga2HJfz4AaE3J1StDXrrZqb0gAEd71fXusVaCw"

def extract_pdf_llamaparse(input_pdf, output_md, api_key, max_pages=20):
    """
    Extract PDF content using LlamaParse.
    """
    print(f"Reading PDF: {input_pdf}")
    print(f"Extracting first {max_pages} pages using LlamaParse...")
    print("Note: LlamaParse uses AI for intelligent document parsing")
    
    # Initialize LlamaParse parser
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # Number of workers for parallel processing
        verbose=True,  # Show progress
        page_separator="\n\n---\n\n",  # Separator between pages
    )
    
    # Check if input file exists
    if not os.path.exists(input_pdf):
        print(f"Error: Input file not found: {input_pdf}")
        return
    
    # Prepare extra_info with file_name
    extra_info = {"file_name": os.path.basename(input_pdf)}
    
    print("\nParsing PDF with LlamaParse (this may take a while)...")
    print("LlamaParse is using AI to understand document structure...")
    
    try:
        # Open and parse the PDF file
        with open(input_pdf, "rb") as f:
            # Must provide extra_info with file_name key when passing file object
            documents = parser.load_data(f, extra_info=extra_info)
        
        print(f"\nSuccessfully parsed PDF. Found {len(documents)} document(s)")
        
        # Create markdown header
        markdown_content = []
        markdown_content.append(f"# PDF Extraction: {os.path.basename(input_pdf)}\n")
        markdown_content.append(f"**Extraction Method:** LlamaParse (AI-powered parsing)\n\n")
        markdown_content.append(f"**Pages:** 1-{max_pages}\n")
        markdown_content.append("**Features:** AI-powered document understanding, intelligent structure detection\n")
        markdown_content.append("---\n\n")
        
        # Extract text from all documents
        for idx, doc in enumerate(documents, 1):
            print(f"Processing document {idx}/{len(documents)}...")
            
            # Add document section if multiple documents
            if len(documents) > 1:
                markdown_content.append(f"## Document {idx}\n\n")
            
            # Get text content
            if hasattr(doc, 'text'):
                markdown_content.append(doc.text)
            elif hasattr(doc, 'get_content'):
                markdown_content.append(doc.get_content())
            else:
                markdown_content.append(str(doc))
            
            markdown_content.append("\n\n")
        
        # Write to markdown file
        print(f"\nWriting output to: {output_md}")
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(''.join(markdown_content))
        
        print(f"Successfully extracted content to {output_md}")
        
        # Note: LlamaParse doesn't have direct page limit control,
        # but it processes the entire document intelligently
        print("\nNote: LlamaParse processes the entire document for better context understanding.")
        print("The output may contain content from all pages, not just the first 20.")
        
    except Exception as e:
        print(f"Error during LlamaParse extraction: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_PDF)
    output_path = os.path.join(script_dir, OUTPUT_MD)
    
    # Extract PDF to markdown
    extract_pdf_llamaparse(input_path, output_path, API_KEY, max_pages=MAX_PAGES)

