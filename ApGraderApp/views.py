from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader

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

@csrf_exempt
async def saq_view(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        # Parse and validate questions
        questions = request.POST.get("questions", "").strip()
        if not questions:
            logging.error('Missing "questions" in request.')
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)
        try:
            questions = json.loads(questions)
            logging.debug("Questions parsed successfully.")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON for questions: {e}")
            return JsonResponse({'error': 'Invalid JSON format for "questions"'}, status=400)

        # Parse and validate PDF
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
                logging.error(f"Error processing PDF: {e}")
                return JsonResponse({'error': 'Failed to process PDF file'}, status=500)
        else:
            logging.error("No PDF file provided in request.")
            return JsonResponse({'error': 'PDF file is required'}, status=400)

        # Parse and validate image
        image_data = None
        if 'image' in request.FILES:
            image = request.FILES['image']
            try:
                # Allow more formats now
                if image.content_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                    logging.error(f"Unsupported image type: {image.content_type}")
                    return JsonResponse({'error': 'Unsupported image type. Only JPEG, PNG, GIF, and WebP are allowed.'}, status=400)
                
                # Read image data as bytes
                image_data = image.read()
                if not image_data:
                    logging.error("Uploaded image is empty.")
                    return JsonResponse({'error': 'Uploaded image is empty.'}, status=400)
                logging.debug(f"Image read successfully: {len(image_data)} bytes")
            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return JsonResponse({'error': 'Failed to process image file.'}, status=500)
        else:
            logging.error("No image file provided in request.")
            return JsonResponse({'error': 'Image file is required'}, status=400)

        # Evaluate
        try:
            response = await asyncio.to_thread(evaluate1, questions, essay_text, image_data)
            logging.info(f"Evaluation successful: {response}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        # Return response
        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        logging.error(f"Error in saq_view: {e}")
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)