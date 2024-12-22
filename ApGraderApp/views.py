from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
from .updatedai import evaluate
from asgiref.sync import sync_to_async
import logging
from io import BytesIO
import json

logger = logging.getLogger(__name__)


@csrf_exempt
async def process(request):
    try:
        if request.method == "POST":
            json_data = None
            file_data = None

            
            if 'file' not in request.FILES:
                return JsonResponse({'error': 'No file provided'}, status=400)

            pdf_file = request.FILES['file']

            
            pdf_stream = BytesIO(pdf_file.read())
            reader = PdfReader(pdf_stream)
            file_data = "".join([page.extract_text() for page in reader.pages])
            logger.info("Completed PDF text extraction")

          
            if request.body:
                try:
                    json_data = json.loads(request.body)
                    logger.info(f"Received JSON data: {json_data}")
                except json.JSONDecodeError:
                    return JsonResponse({'error': 'Invalid JSON data'}, status=400)

            
            if file_data and json_data:
                prompt = json_data.get("prompt")
                response = await sync_to_async(evaluate)(prompt, file_data)
                return JsonResponse({'response': response}, status=200)

            elif file_data:
                return JsonResponse({'error': 'JSON data is required to evaluate the essay'}, status=400)

            elif json_data:
                return JsonResponse({'error': 'PDF file is required for evaluation'}, status=400)

            else:
                return JsonResponse({'error': 'No file or JSON data provided'}, status=400)

        else:
            return JsonResponse({'error': 'Method not allowed'}, status=405)

    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)



