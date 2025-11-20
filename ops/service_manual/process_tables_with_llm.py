"""
Script to process HTML tables in markdown file through LLM and replace with formatted text.

This script:
1. Reads the markdown file
2. Finds all HTML table tags
3. Processes each table through LLM to get semantic text representation
4. Replaces the original HTML table with the LLM response
"""

import logging
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("process_tables_with_llm")

# Load environment variables
load_dotenv()

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# File paths
INPUT_FILE = os.path.join(SCRIPT_DIR, "F115AET_68V_28197_ZA_C1_english_only.md")
OUTPUT_FILE = INPUT_FILE  # Replace in place

# LLM Configuration
LLM_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def find_html_tables(content: str) -> List[Tuple[int, int, str]]:
    """
    Find all HTML table tags in the content.
    
    Returns:
        List of tuples: (start_index, end_index, table_html)
    """
    tables = []
    pattern = r'<table>.*?</table>'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        start = match.start()
        end = match.end()
        table_html = match.group(0)
        tables.append((start, end, table_html))
    
    logger.info(f"Found {len(tables)} HTML tables in the file")
    return tables


def process_table_with_llm(client: OpenAI, table_html: str, table_num: int, total_tables: int) -> str:
    """
    Process an HTML table through LLM to get semantic text representation.
    
    Args:
        client: OpenAI client
        table_html: The HTML table content
        table_num: Current table number (for logging)
        total_tables: Total number of tables (for logging)
    
    Returns:
        Formatted text representation of the table
    """
    logger.info(f"Processing table {table_num}/{total_tables}...")
    
    prompt = (
        "You are a technical documentation assistant for marine engine service manuals.\n\n"
        "Process the HTML table below into a semantically meaningful textual representation.\n"
        "Your task is to:\n"
        "1. Extract all data from the HTML table structure\n"
        "2. Organize it in a clear, readable format\n"
        "3. Preserve all technical information, specifications, and relationships\n"
        "4. Use appropriate formatting (headers, lists, sections) to make it easy to read\n"
        "5. Maintain technical terminology and units exactly as they appear\n"
        "6. Group related information logically\n\n"
        "Respond with plain text only (no markdown formatting, no HTML).\n"
        "Make the output clear and structured so it can be easily understood and searched.\n\n"
        f"HTML Table:\n{table_html}"
    )
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical documentation assistant specializing in marine engine service manuals. You convert HTML tables into clear, structured text while preserving all technical details."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
        )
        
        formatted_text = response.choices[0].message.content.strip()
        logger.info(f"✓ Successfully processed table {table_num}/{total_tables}")
        return formatted_text
        
    except Exception as e:
        logger.error(f"✗ Error processing table {table_num}: {e}")
        # Return a fallback representation
        return f"[Table {table_num} - Error processing: {str(e)}]"


def replace_tables_in_content(content: str, tables: List[Tuple[int, int, str]], client: OpenAI) -> str:
    """
    Replace HTML tables in content with LLM-processed text.
    
    Args:
        content: Original file content
        tables: List of (start, end, table_html) tuples
        client: OpenAI client
    
    Returns:
        Content with tables replaced
    """
    # Process tables in reverse order to maintain correct indices
    processed_content = content
    total_tables = len(tables)
    
    for idx, (start, end, table_html) in enumerate(reversed(tables), 1):
        table_num = total_tables - idx + 1
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing table {table_num}/{total_tables}")
        logger.info(f"{'='*80}")
        
        # Process table through LLM
        formatted_text = process_table_with_llm(client, table_html, table_num, total_tables)
        
        # Replace the table HTML with formatted text
        # Add some spacing for readability
        replacement = f"\n\n{formatted_text}\n\n"
        processed_content = processed_content[:start] + replacement + processed_content[end:]
        
        logger.info(f"Replaced table {table_num} in content")
    
    return processed_content


def main():
    """Main function to process tables in markdown file."""
    logger.info("=" * 80)
    logger.info("HTML TABLE PROCESSING WITH LLM")
    logger.info("=" * 80)
    
    # Validate environment
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # Read the markdown file
    logger.info(f"Reading markdown file: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Read {len(content)} characters from file")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return
    
    # Find all HTML tables
    tables = find_html_tables(content)
    
    if not tables:
        logger.info("No HTML tables found in the file. Exiting.")
        return
    
    # Initialize OpenAI client
    logger.info("Initializing OpenAI client...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Process and replace tables
    logger.info(f"\nProcessing {len(tables)} tables through LLM...")
    processed_content = replace_tables_in_content(content, tables, client)
    
    # Write the processed content back to file
    logger.info(f"\nWriting processed content to: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        logger.info("✓ Successfully wrote processed content to file")
    except Exception as e:
        logger.error(f"Failed to write file: {e}")
        return
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total tables processed: {len(tables)}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

