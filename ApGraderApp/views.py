from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from PyPDF2 import PdfReader
from .AI import evaluate_essay
from asgiref.sync import sync_to_async
import fitz
import aiofiles
import logging
logger = logging.getLogger(__name__)

@csrf_exempt
async def process(request):
    try:
        if request.method == "POST":
            if 'file' not in request.FILES:
                return JsonResponse({'error': 'No file provided'}, status=400)

            pdf_file = request.FILES['file']
            async with aiofiles.open(pdf_file.temporary_file_path(), mode='rb') as f:
                data = await f.read()
            # Ensure the file is a readable PDF
            doc = fitz.open(stream=data, filetype="pdf")
            student_essay = "".join([page.get_text() for page in doc])
            logger.info("Completed PDF text extraction")
            # Process the essay
            response = await sync_to_async(evaluate_essay)(student_essay)
            return JsonResponse({'response': response}, status=200)

        return JsonResponse({'error': 'Invalid request method'}, status=405)
    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)


