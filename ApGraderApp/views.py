from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
import base64
from asgiref.sync import sync_to_async
import logging
from io import BytesIO
import json
from .ApushLEQ import evaluate  
from .ApushSAQ import evaluate1
import io
import asyncio
logger = logging.getLogger(__name__)
from PIL import Image
@csrf_exempt
async def ApushLEQ(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        
        if 'file' in request.FILES:
            pdf_file = request.FILES['file']
            try:
                pdf_stream = BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                return JsonResponse({'error': 'Failed to process the PDF file'}, status=500)
        else:
            return JsonResponse({'error': 'PDF file is required'}, status=400)

        
        try:
            response = await sync_to_async(evaluate)(prompt, essay_text)
            logging.info(f"Evaluation successful: {response}")
        except ValueError as e:
            logger.error(f"Evaluation failed: {e}")
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        
        return JsonResponse({
            "response": {
                "output": response
            }
        }, status=200)

    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({
            "error": "Internal Server Error",
            "details": str(e)
        }, status=500)

import logging

@csrf_exempt
async def saq_view(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        logging.debug("Parsing 'questions' parameter...")
        # Parse and validate questions
        questions = request.POST.get("questions", "").strip()
        if not questions:
            logging.error("Missing 'questions' parameter.")
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)
        try:
            questions = json.loads(questions)
            logging.debug("Parsed 'questions' successfully.")
        except json.JSONDecodeError:
            logging.error("Invalid JSON in 'questions'.")
            return JsonResponse({'error': 'Invalid JSON format for "questions"'}, status=400)

        # Parse and validate PDF
        logging.debug("Checking for PDF file...")
        if 'essay_file' in request.FILES:
            pdf_file = request.FILES['essay_file']
            try:
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if not essay_text.strip():
                    logging.warning("PDF contains no extractable text.")
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
                logging.debug("PDF processed successfully.")
            except Exception as e:
                logging.error(f"Error processing PDF file: {e}")
                return JsonResponse({'error': 'Failed to process PDF file'}, status=500)
        else:
            logging.error("PDF file not provided.")
            return JsonResponse({'error': 'PDF file is required'}, status=400)

        # Parse and validate Base64 image
        logging.debug("Checking for image file...")
        if 'image' in request.FILES:
            image = request.FILES['image']
            supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            if image.content_type not in supported_mime_types:
                logging.error(f"Unsupported image type: {image.content_type}")
                return JsonResponse({'error': 'Unsupported image type.'}, status=400)

            try:
                image_data = base64.b64encode(image.read()).decode('utf-8')  # Encode to Base64
                logging.debug(f"Image processed successfully. Size: {len(image_data)} bytes.")
            except Exception as e:
                logging.error(f"Error processing image file: {e}")
                return JsonResponse({'error': 'Failed to process image file.'}, status=500)
        else:
            logging.error("Image file not provided.")
            return JsonResponse({'error': 'Image file is required'}, status=400)

        # Evaluate
        logging.debug("Calling 'evaluate1'...")
        try:
            response = await asyncio.to_thread(evaluate1, questions, essay_text, image_data)
            logging.info("Evaluation completed successfully.")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        # Return response
        logging.debug("Returning response.")
        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
