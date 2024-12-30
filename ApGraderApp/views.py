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
from .ApushDBQ import evaluate2
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
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either "essay_file" or "essay_text" is required'}, status=400)
        
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
        
        questions = request.POST.get("questions", "").strip()
        if not questions:
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)

        try:
            questions = json.loads(questions)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format for "questions"'}, status=400)

        
        essay_text = None
        if 'essay_file' in request.FILES:
            pdf_file = request.FILES['essay_file']
            try:
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception:
                return JsonResponse({'error': 'Failed to process PDF file'}, status=500)
        else:
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either "essay_file" or "essay_text" is required'}, status=400)

       
        image_data = None
        if 'image' in request.FILES:
            image = request.FILES['image']
            supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            if image.content_type not in supported_mime_types:
                return JsonResponse({'error': 'Unsupported image type.'}, status=400)

            try:
                image_data = base64.b64encode(image.read()).decode('utf-8')
            except Exception:
                return JsonResponse({'error': 'Failed to process image file.'}, status=500)

       
        try:
            response = await sync_to_async(evaluate1)(questions, essay_text,image_data)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        
        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    
@csrf_exempt
async def dbq_view(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        
        if 'essay_file' not in request.FILES:
            return JsonResponse({'error': 'PDF file is required'}, status=400)

        try:
            pdf_file = request.FILES['essay_file']
            pdf_stream = io.BytesIO(pdf_file.read())
            reader = PdfReader(pdf_stream)
            essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            if not essay_text.strip():
                return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
        except Exception:
            return JsonResponse({'error': 'Failed to process PDF file'}, status=500)

        else:
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either "essay_file" or "essay_text" is required'}, status=400)
        
        images = []
        for i in range(1, 8):  
            image_key = f'image{i}'
            if image_key in request.FILES:
                image = request.FILES[image_key]
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {image_key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception:
                    return JsonResponse({'error': f'Failed to process {image_key}.'}, status=500)

        
        images = images[:7] + [None] * (7 - len(images))

        
        try:
            response = await sync_to_async(evaluate2)(prompt, essay_text, images)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    
@csrf_exempt
def bulk_grading(request):
    pass