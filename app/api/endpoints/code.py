from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

from app.services.code_agent import CodeAgent
from app.api.dependencies import get_agent
from app.schemas import CodeUploadResponse, CodeProcessResponse, CodeSearchQuery

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload", response_model=CodeUploadResponse)
async def upload_code_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("python"),
    agent: CodeAgent = Depends(get_agent)
):
    """
    Upload a code file to be indexed in the vector database.
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Schedule the background task to store code in Qdrant
        background_tasks.add_task(
            store_code_in_background, agent, str(file_path), content, language
        )
        
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded and processing started",
            "file_name": file.filename,
            "language": language
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def store_code_in_background(
    agent: CodeAgent, file_path: str, content: str, language: str
):
    """Background task to store code in the vector database"""
    try:
        result = await agent.store_code(file_path, content, language)
        logger.info(f"Background code storage completed: {result}")
    except Exception as e:
        logger.error(f"Error in background code storage: {str(e)}", exc_info=True)

@router.post("/process-directory", response_model=CodeProcessResponse)
async def process_code_directory(
    background_tasks: BackgroundTasks,
    directory: str = Form(...),
    languages: List[str] = Form(["py", "js", "cs", "java", "cpp", "c", "go", "rs"]),
    agent: CodeAgent = Depends(get_agent)
):
    """
    Process all code files in a directory and index them in the vector database.
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory {directory} does not exist")
        
        # Schedule background task to process all files in the directory
        background_tasks.add_task(
            process_directory_in_background, agent, str(dir_path), languages
        )
        
        return {
            "status": "success",
            "message": f"Started processing directory: {directory}",
            "directory": directory,
            "supported_languages": languages
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing directory: {str(e)}")

async def process_directory_in_background(
    agent: CodeAgent, directory: str, languages: List[str]
):
    """Background task to process all code files in a directory"""
    try:
        dir_path = Path(directory)
        files_processed = 0
        
        # Map file extensions to language names
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "cs": "csharp",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rs": "rust",
            "html": "html",
            "css": "css",
            "php": "php",
            "rb": "ruby",
            "kt": "kotlin",
            "swift": "swift",
        }
        
        # Process all files with supported extensions
        for ext in languages:
            for file_path in dir_path.glob(f"**/*.{ext}"):
                try:
                    # Skip files in node_modules, __pycache__, .git, etc.
                    if any(part.startswith((".", "__")) for part in file_path.parts):
                        continue
                    
                    # Read file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    # Store code in vector database
                    language = lang_map.get(ext, ext)
                    await agent.store_code(str(file_path), content, language)
                    files_processed += 1
                    
                    logger.info(f"Processed file: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
        
        logger.info(f"Directory processing completed. Processed {files_processed} files.")
    except Exception as e:
        logger.error(f"Error in background directory processing: {str(e)}", exc_info=True)

@router.post("/search", response_model=Dict[str, Any])
async def search_code(
    query: CodeSearchQuery,
    agent: CodeAgent = Depends(get_agent)
):
    """
    Search for code snippets related to the query.
    """
    try:
        # Custom search implementation that uses vector search directly
        # This is different from the full agent run which also generates fixes
        result = await agent.run(query.query)
        return {
            "query": query.query,
            "results": result.get("code_snippets", []),
            "total": len(result.get("code_snippets", []))
        }
    except Exception as e:
        logger.error(f"Error searching code: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching code: {str(e)}") 