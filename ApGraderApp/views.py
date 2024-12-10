from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
from .AI import evaluate_essay
from asgiref.sync import sync_to_async
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

            # Read the file asynchronously as a binary stream
            data = pdf_file.read()  # Use `read` to access InMemoryUploadedFile content

            # Ensure the file is a readable PDF
            reader = PdfReader()
            reader.append(data)

            student_essay = "".join([page.extract_text() for page in reader.pages])
            logger.info("Completed PDF text extraction")

            # Process the essay asynchronously
            response = await sync_to_async(evaluate_essay)(student_essay)

            return JsonResponse({'response': response}, status=200)

        return JsonResponse({'error': 'Invalid request method'}, status=405)
    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
