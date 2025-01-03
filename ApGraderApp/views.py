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
from .ApEuroLEQ import euro_leq_bulk
from .ApushSAQ import evaluate1
from .ApushDBQ import evaluate2
import io
import asyncio
logger = logging.getLogger(__name__)
from PIL import Image
from .ApEuroLEQ import evaluateeuroleq
from .ApEuroSAQ import evaluateeurosaq
from .ApEuroDBQ import evaluateeurodbq
from .ApEuroSAQ import euro_saq_bulk_grading
import zipfile
from django.http import HttpResponse
from .ApEuroDBQ import evaluateeurodbqbulk
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

        
        essay_text = None
        if 'essay_file' in request.FILES:
            
            pdf_file = request.FILES['essay_file']
            try:
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = ""
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    essay_text += extracted
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception:
                return JsonResponse({'error': 'Failed to process PDF file'}, status=500)
        else:
            
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({
                    'error': 'Either "essay_file" or "essay_text" is required'
                }, status=400)

        
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
            
            response = await sync_to_async(evaluate1)(questions, essay_text, image_data)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    

    
from asgiref.sync import sync_to_async

@csrf_exempt
async def bulk_grading_leq(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        prompt = request.POST.get('prompt', '').strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    
                    response = await sync_to_async(euro_leq_bulk)(prompt, image_data)
                    file_name = f"{file.name}_response.txt"
                    zip_file.writestr(file_name, response)
                except Exception as e:
                    return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response
    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    
                

@csrf_exempt
async def euro_saq_bulk(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        questions = request.POST.get('questions', '').strip()
        if not questions:
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        stim_data = request.FILES.get('stimulus', None)
        if stim_data:

            stim_data = base64.b64encode(stim_data.read()).decode('utf-8')
        else:
            stim_data = None
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    response = await sync_to_async(euro_saq_bulk_grading)(questions, image_data, stim_data)
                    file_name = f"{file.name}_response.txt"
                    zip_file.writestr(file_name, response)
                except Exception as e:
                    return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response
    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
async def euro_dbq_bulk(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        images = []
        for key in request.FILES:
            if key.startswith('image_'):
                image = request.FILES[key]
                supported_mime_types = ["image/jpeg", "image/png", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {key}: {str(e)}'}, status=500)

        essays = request.FILES.getlist('essays')
        if not essays:
            return JsonResponse({'error': 'No essays provided.'}, status=400)

        
        allowed_image_mime_types = ["image/jpeg", "image/png", "image/webp"]

        for essay in essays:
            if essay.content_type not in allowed_image_mime_types:
                return JsonResponse({
                    'error': f'Unsupported file type for essay "{essay.name}". '
                             f'Allowed types: {allowed_image_mime_types}'
                }, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for essay in essays:
                try:
                    essay_data = base64.b64encode(essay.read()).decode('utf-8')
                    if not essay_data:
                        return JsonResponse({'error': f'Empty or unreadable essay: {essay.name}'}, status=400)

                    
                    response_text = await sync_to_async(evaluateeurodbqbulk)(prompt, essay_data, images)
                    file_name = f"{essay.name}_response.txt"
                    zip_file.writestr(file_name, response_text)
                except Exception as e:
                    return JsonResponse({
                        'error': f'Evaluation failed for {essay.name}',
                        'details': str(e)
                    }, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
from .ApushDBQ import evaluate22
@csrf_exempt

async def apushdbqbulk(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        images = []
        for key in request.FILES:
            if key.startswith('image_'):
                image = request.FILES[key]
                supported_mime_types = ["image/jpeg", "image/png", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {key}: {str(e)}'}, status=500)

        essays = request.FILES.getlist('essays')
        if not essays:
            return JsonResponse({'error': 'No essays provided.'}, status=400)

        
        allowed_image_mime_types = ["image/jpeg", "image/png", "image/webp"]

        for essay in essays:
            if essay.content_type not in allowed_image_mime_types:
                return JsonResponse({
                    'error': f'Unsupported file type for essay "{essay.name}". '
                             f'Allowed types: {allowed_image_mime_types}'
                }, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for essay in essays:
                try:
                    essay_data = base64.b64encode(essay.read()).decode('utf-8')
                    if not essay_data:
                        return JsonResponse({'error': f'Empty or unreadable essay: {essay.name}'}, status=400)

                   
                    response_text = await sync_to_async(evaluate22)(prompt, essay_data, images)
                    file_name = f"{essay.name}_response.txt"
                    zip_file.writestr(file_name, response_text)
                except Exception as e:
                    return JsonResponse({
                        'error': f'Evaluation failed for {essay.name}',
                        'details': str(e)
                    }, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)

@csrf_exempt
async def ApEuroLEQ(request):
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
            response = await sync_to_async(evaluateeuroleq)(prompt, essay_text)
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

from .ApushLEQ import evaluate69    





@csrf_exempt
async def apushleqbulk(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        prompt = request.POST.get('prompt', '').strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    
                    response = await sync_to_async(evaluate69)(prompt, image_data)
                    file_name = f"{file.name}_response.txt"
                    zip_file.writestr(file_name, response)
                except Exception as e:
                    return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response
    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    

from .ApushSAQ import evaluate11

@csrf_exempt
async def apushsaqbulk(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        questions = request.POST.get('questions', '').strip()
        if not questions:
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        stim_data = request.FILES.get('stimulus', None)
        if stim_data:

            stim_data = base64.b64encode(stim_data.read()).decode('utf-8')
        else:
            stim_data = None
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    response = await sync_to_async(evaluate11)(questions, image_data, stim_data)
                    file_name = f"{file.name}_response.txt"
                    zip_file.writestr(file_name, response)
                except Exception as e:
                    return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response
    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)


@csrf_exempt
async def eurosaq_view(request):
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
            response = await sync_to_async(evaluateeurosaq)(questions, essay_text,image_data)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        
        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    
@csrf_exempt
async def eurodbq(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        essay_text = ""
        if 'essays' in request.FILES: 
            try:
                pdf_file = request.FILES['essays']
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'Failed to process PDF file: {str(e)}'}, status=400)
        else:  
            essay_text = request.POST.get("essays", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either a PDF file or essay text is required'}, status=400)

        
        images = []
        for i in range(1, 8):
            image_key = f'image_{i}'
            if image_key in request.FILES:
                image = request.FILES[image_key]
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {image_key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {image_key}: {str(e)}'}, status=500)

        
        images = images[:7] + [None] * (7 - len(images))

        
        try:
            response = await sync_to_async(evaluateeurodbq)(prompt, essay_text, images)
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

        essay_text = ""
        if 'essays' in request.FILES: 
            try:
                pdf_file = request.FILES['essays']
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'Failed to process PDF file: {str(e)}'}, status=400)
        else:  
            essay_text = request.POST.get("essays", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either a PDF file or essay text is required'}, status=400)

        
        images = []
        for i in range(1, 8):
            image_key = f'image_{i}'
            if image_key in request.FILES:
                image = request.FILES[image_key]
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {image_key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {image_key}: {str(e)}'}, status=500)

        
        images = images[:7] + [None] * (7 - len(images))

        
        try:
            response = await sync_to_async(evaluate2)(prompt, essay_text, images)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)