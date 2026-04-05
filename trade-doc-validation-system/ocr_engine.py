"""
OCR Engine Module
Converts PDF documents to images and extracts raw text using Tesseract OCR
Handles both native PDFs and scanned documents
"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """
    Handles OCR processing for PDF documents.
    Converts PDFs to images and extracts text using Tesseract OCR.
    """
    
    def __init__(self, dpi: int = 300, timeout: int = 60):
        """
        Initialize OCR Engine
        
        Args:
            dpi (int): DPI for PDF to image conversion (default: 300)
            timeout (int): Timeout for OCR processing in seconds
        """
        self.dpi = dpi
        self.timeout = timeout
        self._verify_tesseract()
    
    def _verify_tesseract(self) -> None:
        """Verify Tesseract OCR is installed and accessible"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.warning(f"Tesseract not found: {e}. Install via: apt-get install tesseract-ocr")
    
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF document to list of PIL Image objects
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Image.Image]: List of PIL Image objects, one per page
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If conversion fails
        """
        try:
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(pdf_path, dpi=self.dpi)
            logger.info(f"Successfully converted {len(images)} pages from PDF")
            return images
        
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    def pdf_to_bytes(self, pdf_path: str) -> List[bytes]:
        """
        Convert PDF pages to bytes for processing
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[bytes]: List of image bytes (PNG format)
        """
        try:
            images = self.pdf_to_images(pdf_path)
            image_bytes_list = []
            
            for idx, image in enumerate(images):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes_list.append(img_byte_arr.getvalue())
                logger.debug(f"Page {idx + 1} converted to bytes")
            
            return image_bytes_list
        
        except Exception as e:
            logger.error(f"Error converting PDF to bytes: {str(e)}")
            raise
    
    def extract_text(self, pdf_path: str, lang: str = 'eng') -> Dict[int, str]:
        """
        Extract raw text from PDF document using OCR
        
        Args:
            pdf_path (str): Path to the PDF file
            lang (str): Tesseract language code (default: 'eng' for English)
            
        Returns:
            Dict[int, str]: Dictionary mapping page number to extracted text
                           Keys are 0-indexed page numbers
        """
        try:
            images = self.pdf_to_images(pdf_path)
            extracted_data = {}
            
            logger.info(f"Starting OCR extraction with language: {lang}")
            
            for page_num, image in enumerate(images):
                try:
                    # Convert image to RGB if necessary (handles RGBA, grayscale, etc.)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Perform OCR
                    raw_text = pytesseract.image_to_string(image, lang=lang)
                    extracted_data[page_num] = raw_text
                    
                    logger.info(f"Extracted {len(raw_text)} characters from page {page_num + 1}")
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    extracted_data[page_num] = ""
            
            return extracted_data
        
        except Exception as e:
            logger.error(f"Error in extract_text: {str(e)}")
            raise
    
    def extract_text_with_confidence(self, pdf_path: str, lang: str = 'eng') -> Dict[int, Dict[str, any]]:
        """
        Extract text with confidence scores from OCR
        
        Args:
            pdf_path (str): Path to the PDF file
            lang (str): Tesseract language code
            
        Returns:
            Dict[int, Dict]: Page data including text and average confidence score
        """
        try:
            images = self.pdf_to_images(pdf_path)
            extracted_data = {}
            
            for page_num, image in enumerate(images):
                try:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Extract text
                    text = pytesseract.image_to_string(image, lang=lang)
                    
                    # Get detailed data with confidence scores
                    data = pytesseract.image_to_data(image, lang=lang, output_type='dict')
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    extracted_data[page_num] = {
                        'text': text,
                        'confidence': avg_confidence,
                        'word_count': len(text.split()),
                        'character_count': len(text)
                    }
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    extracted_data[page_num] = {
                        'text': '',
                        'confidence': 0,
                        'word_count': 0,
                        'character_count': 0,
                        'error': str(e)
                    }
            
            return extracted_data
        
        except Exception as e:
            logger.error(f"Error in extract_text_with_confidence: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image, enhance: bool = True) -> Image.Image:
        """
        Preprocess image for better OCR results
        (Optional: improves accuracy for low-quality documents)
        
        Args:
            image (Image.Image): PIL Image object
            enhance (bool): Whether to apply enhancement filters
            
        Returns:
            Image.Image: Preprocessed image
        """
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if enhance:
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)
                
                # Apply slight blur to reduce noise
                image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            return image
        
        except Exception as e:
            logger.warning(f"Error in image preprocessing: {str(e)}")
            return image
    
    def extract_text_enhanced(self, pdf_path: str, lang: str = 'eng', 
                             preprocess: bool = True) -> Dict[int, str]:
        """
        Extract text with image preprocessing for better accuracy
        
        Args:
            pdf_path (str): Path to the PDF file
            lang (str): Tesseract language code
            preprocess (bool): Whether to preprocess images
            
        Returns:
            Dict[int, str]: Dictionary of page number to extracted text
        """
        try:
            images = self.pdf_to_images(pdf_path)
            extracted_data = {}
            
            for page_num, image in enumerate(images):
                try:
                    if preprocess:
                        image = self.preprocess_image(image, enhance=True)
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    text = pytesseract.image_to_string(image, lang=lang)
                    extracted_data[page_num] = text
                    logger.info(f"Extracted {len(text)} chars from page {page_num + 1}")
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    extracted_data[page_num] = ""
            
            return extracted_data
        
        except Exception as e:
            logger.error(f"Error in extract_text_enhanced: {str(e)}")
            raise
    
    def extract_by_region(self, pdf_path: str, page_num: int, 
                         bbox: Tuple[int, int, int, int], lang: str = 'eng') -> str:
        """
        Extract text from a specific region of a PDF page
        Useful for extracting specific fields from documents
        
        Args:
            pdf_path (str): Path to the PDF file
            page_num (int): Page number (0-indexed)
            bbox (tuple): Bounding box (left, top, right, bottom) in pixels
            lang (str): Tesseract language code
            
        Returns:
            str: Extracted text from the region
        """
        try:
            images = self.pdf_to_images(pdf_path)
            
            if page_num >= len(images):
                raise ValueError(f"Page {page_num} not found in PDF")
            
            image = images[page_num]
            cropped = image.crop(bbox)
            
            if cropped.mode != 'RGB':
                cropped = cropped.convert('RGB')
            
            text = pytesseract.image_to_string(cropped, lang=lang)
            logger.info(f"Extracted text from region: {len(text)} characters")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text by region: {str(e)}")
            raise
    
    def batch_process_pdfs(self, pdf_paths: List[str], lang: str = 'eng') -> Dict[str, Dict[int, str]]:
        """
        Process multiple PDF files
        
        Args:
            pdf_paths (List[str]): List of PDF file paths
            lang (str): Tesseract language code
            
        Returns:
            Dict[str, Dict[int, str]]: Dictionary mapping file names to extracted text
        """
        results = {}
        
        for pdf_path in pdf_paths:
            try:
                filename = Path(pdf_path).name
                logger.info(f"Processing batch file: {filename}")
                results[filename] = self.extract_text(pdf_path, lang=lang)
            
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                results[Path(pdf_path).name] = {'error': str(e)}
        
        return results


# Utility function for easy access
def get_ocr_engine(dpi: int = 300) -> OCREngine:
    """Factory function to create OCR engine instance"""
    return OCREngine(dpi=dpi)
